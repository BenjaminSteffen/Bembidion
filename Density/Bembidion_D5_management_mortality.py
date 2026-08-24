"""
Density-dependence variant for the Bembidion modelling project.

Movement basis
--------------
The DIRECTION_SEARCH_OFFSETS constant and move_to() implementation below are copied from
Bombidion_Movment(7).py without the M1-M4 movement modifications. Density-dependent logic is
implemented separately after the movement function so that movement behaviour remains unchanged.

Population compatibility
------------------------
The added functions operate on Polars DataFrames and support the ``lifestage`` convention used in
Population.py. If no ``lifestage`` column exists, the original movement-only population is treated
as adults where that is meaningful.

IMPORTANT: Parameters labelled PROVISIONAL are scenario/calibration parameters, not empirically
validated Bembidion lampros estimates.
"""

import math
import polars_random
import polars as pl

# -----------------------------------------------------------------------------
# ORIGINAL MOVEMENT IMPLEMENTATION (unchanged)
# -----------------------------------------------------------------------------
import polars_random
import polars as pl

# The set of alternative headings (in degrees) evaluated when an individual's current heading fails. They are
# applied relative to the direction the individual had before the search started this step and cover the entire
# remaining circle (45-degree steps) exactly once each. All seven headings are evaluated simultaneously in Phase 2
# of move_to below, and one of the successful ones is then chosen uniformly at random, so the order of this list
# has no effect on the outcome.
DIRECTION_SEARCH_OFFSETS = [45, -45, 90, -90, 135, -135, 180]


def move_to(
        individuals: pl.DataFrame,
        landscape: pl.DataFrame,
        max_distance: int,
        turning_probability: float
) -> pl.DataFrame:
    """
    This function implements the daily movement of dispersing bembidions. In its initial form, it should be functionally
    equal to the ALMaSS function Bembidion_Adult::MoveTo. However, instead of specifying the movement for an individual
    bembidion, it operates on the entire population of bembidions passed to the function.

    Args:
        individuals: A polars dataframe representing the individuals to move. Each row of the dataframe represents one
            individual. The dataframe must have at least three columns characterizing the individual: a column x with
            the current x-coordinate of the individual, a column y with its current y-coordinate and a column direction
            with the last movement direction of the individual. Latter can be one of eight values: 0, 45, 90, 135, 180,
            225, 270, or 315.
        landscape: A polars dataframe containing information about the accessibility of patches for bembidions. The
            dataframe has at least three columns: a column x with the integer x-coordinate of the one square
            meter-patch, a column y with its y-coordinate, and a column accessibility with one of the three string
            literals full, partial or none indicating the accessibility of the patch for the bembidions.
        max_distance: The maximum distance each individual will move in steps. A step can be either one meter in
            horizontal or vertical direction or sqrt(1) m in diagonal direction. The actual number of steps of each
            bembidion is randomly sampled from a uniform distribution. The lower bound of movement is always one step.
        turning_probability: A value between 0 and 1 indicating the probability by which a bembidion changes its
            current direction. Clockwise and counterclockwise turns always have the same probability of one half of
            the turning probability.

    Returns:
        A Polars dataframe with the same schema as the individuals argument, but with updated coordinates and direction
        of bembidions.
    """
    # We have to initially prepare additiona state variables defining the movement of the individuals.
    individuals = (

        # Our starting point is the population of individuals passed to the function.
        individuals

        # The first thing a bembidion does in the ALMaSS code prior movement is determining how many steps it does. We
        # add this information by randomly drawing from a uniform distribution with max_distance as its upper bound.
        .with_columns(

            # We add their desired movement distance in steps. The upper bound is exlcusive, so we have to add 1 to
            # include it as a possible sampling result.
            distance=polars_random.randint(1, max_distance + 1),

            # We draw a random number between 0 and 1 that represents the random turning behavior of each bembidion.
            turning=polars_random.uniform()
        )

        # Before movement, the ALMaSS code checks whether a bembidion changes its direction, which has a probability
        # defined by the turning_probability. The probability is symmetrical for clockwise and counterclockwise turns.
        # We do the same here.
        .with_columns(
            direction=(
                pl
                .when(pl.col("turning") < turning_probability / 2).then(pl.col("direction") - 45)
                .when(pl.col("turning") < turning_probability).then(pl.col("direction") + 45)
                .otherwise("direction")
            )
        )

        # After turning, it might be that directions are no longer encoded by the eight allowed values only. This needs
        # to be corrected.
        .with_columns(
            direction=pl.when(direction=-45).then(315).when(direction=360).then(0).otherwise("direction")
        )

        # The turning column is no longer needed.
        .drop("turning")
    )

    # In this list, we collect all individuals that have finished their movement.
    individuals_after_movement = []

    # Bembidions do one step after each other.
    for i in range(max_distance):
        # We output the current step for better development feedback.
        print(f"Movement step {i + 1}")

        # Stop early if there are no more individuals wanting to move
        if individuals is None:
            break

        # As a first action for each step, we partition the individuals according to their need to move further.
        movement = individuals.with_columns(step_done=pl.col("distance").le(i)).partition_by("step_done", as_dict=True)

        # There up to two different groups: individuals that still need to move and those that are finished moving.
        movement_finished = movement.get((True,))
        movement_required = movement.get((False,))

        # If there are individuals with finished movement, their current state is taken as their new state after
        # movement.
        if movement_finished is not None:
            individuals_after_movement.append(movement_finished.drop("step_done", "distance"))

        # For individuals that still need to move this step, we add a synthetic row id. It is used below to group
        # several candidate headings evaluated for the same individual back together when picking an escape
        # direction after a failed movement attempt.
        if movement_required is not None:
            movement_required = movement_required.with_row_index("individual_id")

        # In this list, we collect the results of this step.
        step_results = []

        # Each individual first tries to keep moving in its current heading (Phase 1). Individuals for whom that
        # fails go through a vectorized search step (Phase 2) that evaluates all seven remaining headings at once
        # and picks uniformly at random among whichever of them succeed, instead of retrying with a random turn.
        if movement_required is not None:

            # --- Phase 1: try the individual's current heading. ---
            trial_results = (

                # The starting point of each movement step is always the current state of all individuals requiring
                # movement.
                movement_required

                # The current positions of the bembidions and their movement directions have to be translated into
                # target coordinates of the movement.
                .with_columns(

                    # The x-coordinate is straightforward.
                    new_x=pl
                    .when(pl.col("direction").is_in((225, 270, 315))).then(pl.col("x") - 1)
                    .when(pl.col("direction").is_in((45, 90, 135))).then(pl.col("x") + 1)
                    .otherwise("x"),

                    # For the y-coordinate, it should be considered that in the Northern Hemisphere coordinates increase
                    # in the direction of North.
                    new_y=pl
                    .when(pl.col("direction").is_in((315, 0, 45))).then(pl.col("y") + 1)
                    .when(pl.col("direction").is_in((225, 180, 135))).then(pl.col("y") - 1)
                    .otherwise("y"),

                    # We add also a random number that indicates the decision of a bembidion to access an only partially
                    # accessible patch.
                    access=polars_random.uniform()
                )

                # We can now learn about the accessibility of the target patch by joining the bembidion and the movement
                # map dataframes by coordinates. We need a left join so that we also keep bembidions that move outside
                # the defined landscape.
                .join(landscape, left_on=("new_x", "new_y"), right_on=("x", "y"), how="left")

                # We can now check whether the intended movement is successful.
                .with_columns(
                    success=(
                            pl.col("accessibility").eq("full") |
                            (pl.col("accessibility").eq("partial") & pl.col("access").lt(.4))
                    )
                )
                .partition_by("success", as_dict=True)
            )

            # There are up to three groups of movement trial results: individuals successful with successful movement,
            # individuals with unsuccessful movements and individuals that want to move outside the defined landscape.
            successful_trial = trial_results.get((True,))
            unsuccessful_trial = trial_results.get((False,))
            leaving_map = trial_results.get((None,))

            # Individuals with successful movement update their coordinates and are added to the list of step results.
            if successful_trial is not None:
                step_results.append(
                    successful_trial.select(x="new_x", y="new_y", direction="direction", distance="distance")
                )

            # Individuals for whom Phase 1 failed (destination inaccessible, or outside the landscape) go through
            # Phase 2 below.
            blocked = [df for df in (unsuccessful_trial, leaving_map) if df is not None]
            blocked = pl.concat(blocked) if len(blocked) > 0 else None

            if blocked is not None:

                # --- Phase 2: Alle 7 Ausweichrichtungen gleichzeitig testen ---
                candidates = (
                    blocked
                    .select("individual_id", "x", "y", "direction", "distance")
                    .join(pl.DataFrame({"offset": DIRECTION_SEARCH_OFFSETS}), how="cross")
                    .with_columns(
                        heading=(((pl.col("direction") + pl.col("offset")) % 360) + 360) % 360
                    )
                    .with_columns(
                        new_x=pl.when(pl.col("heading").is_in((225, 270, 315))).then(pl.col("x") - 1)
                                .when(pl.col("heading").is_in((45, 90, 135))).then(pl.col("x") + 1)
                                .otherwise(pl.col("x")),
                        new_y=pl.when(pl.col("heading").is_in((315, 0, 45))).then(pl.col("y") + 1)
                                .when(pl.col("heading").is_in((225, 180, 135))).then(pl.col("y") - 1)
                                .otherwise(pl.col("y")),
                        access=polars_random.uniform(),
                        priority=polars_random.uniform()
                    )
                    .join(landscape, left_on=("new_x", "new_y"), right_on=("x", "y"), how="left")
                    .with_columns(
                        success=(
                            pl.col("accessibility").eq("full") |
                            (pl.col("accessibility").eq("partial") & pl.col("access").lt(.4))
                        ).fill_null(False)  # Verhindert Null-Werte am Kartenrand
                    )
                )
                successful_candidates = candidates.filter(pl.col("success"))
                if not successful_candidates.is_empty():
                    chosen = (
                        successful_candidates
                        .filter(pl.col("priority") == pl.col("priority").max().over("individual_id"))
                        .unique(subset="individual_id", keep="first")
                    )
                    step_results.append(
                        chosen.select(x="new_x", y="new_y", direction="heading", distance="distance")
                    )
                    # Blockierte Käfer ermitteln, die KEINE Ausweichrichtung gefunden haben:
                    stuck = blocked.join(chosen.select("individual_id"), on="individual_id", how="anti")
                else:
                    # Niemand konnte ausweichen -> alle Blockierten stecken fest
                    stuck = blocked
                if not stuck.is_empty():
                    step_results.append(
                        stuck.select("x", "y", "direction", "distance")
                    )

        # Set the individuals for the next step.
        if len(step_results) > 0:
            individuals = pl.concat(step_results)
        else:
            individuals = None

        # Add the last batch of individuals to the result.
        if i == max_distance - 1 and individuals is not None:
            individuals_after_movement.append(individuals.drop("distance"))

    # Return the movement results
    return pl.concat(individuals_after_movement)


# -----------------------------------------------------------------------------
# Shared density helper functions (do not alter movement behaviour)
# -----------------------------------------------------------------------------

def _validate_radius(radius_m: float) -> int:
    """Return an integer cell radius for the 1 m x 1 m landscape grid."""
    if radius_m < 0:
        raise ValueError("radius_m must be >= 0")
    if not float(radius_m).is_integer():
        raise ValueError("radius_m must be an integer number of 1 m cells")
    return int(radius_m)


def _circular_offsets(radius_m: float) -> pl.DataFrame:
    """Integer grid offsets whose cell centres fall within a Euclidean radius."""
    radius = _validate_radius(radius_m)
    offsets = [
        (dx, dy)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        if dx * dx + dy * dy <= radius * radius
    ]
    return pl.DataFrame(
        {
            "_dd_dx": [p[0] for p in offsets],
            "_dd_dy": [p[1] for p in offsets],
        }
    )


def _attach_local_other_count(
        focal: pl.DataFrame,
        neighbours: pl.DataFrame,
        radius_m: float,
        output_column: str,
        subtract_self: bool = True,
) -> pl.DataFrame:
    """
    Attach the number of neighbouring individuals within ``radius_m`` of every focal individual.

    Counts are computed from occupied 1 m cells and expanded over a circular kernel. This avoids an
    O(N^2) individual-by-individual distance matrix. ``subtract_self`` should be True whenever every
    focal individual is also present in ``neighbours``.
    """
    if focal.is_empty():
        return focal.with_columns(pl.lit(0, dtype=pl.Int64).alias(output_column))
    if neighbours.is_empty():
        return focal.with_columns(pl.lit(0, dtype=pl.Int64).alias(output_column))

    offsets = _circular_offsets(radius_m)
    cell_counts = neighbours.group_by("x", "y").agg(pl.len().alias("_dd_cell_count"))

    local_field = (
        cell_counts
        .join(offsets, how="cross")
        .with_columns(
            _dd_target_x=pl.col("x") + pl.col("_dd_dx"),
            _dd_target_y=pl.col("y") + pl.col("_dd_dy"),
        )
        .group_by("_dd_target_x", "_dd_target_y")
        .agg(pl.col("_dd_cell_count").sum().alias("_dd_local_count"))
    )

    result = (
        focal
        .join(
            local_field,
            left_on=("x", "y"),
            right_on=("_dd_target_x", "_dd_target_y"),
            how="left",
        )
        .with_columns(pl.col("_dd_local_count").fill_null(0).cast(pl.Int64))
    )

    if subtract_self:
        result = result.with_columns(
            pl.when(pl.col("_dd_local_count") > 0)
            .then(pl.col("_dd_local_count") - 1)
            .otherwise(0)
            .alias(output_column)
        )
    else:
        result = result.with_columns(pl.col("_dd_local_count").alias(output_column))

    return result.drop("_dd_local_count")


def _adult_subset(individuals: pl.DataFrame) -> pl.DataFrame:
    if "lifestage" not in individuals.columns:
        return individuals
    return individuals.filter(pl.col("lifestage") == "Adult")


def _larval_subset(individuals: pl.DataFrame) -> pl.DataFrame:
    if "lifestage" not in individuals.columns:
        return individuals[:0]
    return individuals.filter(pl.col("lifestage").is_in(("Larva", "Larvae", "L1", "L2", "L3")))


# =============================================================================
# D5 - Mortality from field-management events
# =============================================================================
# These are PROVISIONAL fallback rates derived from generic ALMaSS farm-operation insect effects,
# not verified Bembidion-specific mortality estimates. Prefer explicit mortality_probability values
# in the daily management-events dataframe when species/lifestage-specific values are available.
D5_DEFAULT_EVENT_MORTALITY = {
    "autumn_plough": 0.90,
    "spring_plough": 0.90,
    "deep_ploughing": 0.90,
    "stubble_plough": 0.90,
    "stubble_cultivator_heavy": 0.90,
    "autumn_harrow": 0.90,
    "harvest": 0.60,
    "harvest_long": 0.60,
}


def apply_management_mortality(
        individuals: pl.DataFrame,
        management_events: pl.DataFrame,
        default_event_mortality: dict[str, float] | None = None,
) -> pl.DataFrame:
    """
    Apply event-driven mortality to individuals occupying managed cells.

    Required management_events columns:
      - x, y: affected 1 m cell
      - event: event name

    Optional columns:
      - mortality_probability: explicit probability in [0,1], preferred over fallback event rates
      - lifestage: if present, event only applies to that lifestage

    Multiple events affecting the same individual on one day are combined assuming independent
    survival: p_total = 1 - product(1 - p_event).
    """
    required = {"x", "y", "event"}
    missing = required.difference(management_events.columns)
    if missing:
        raise ValueError(f"management_events is missing required columns: {sorted(missing)}")
    if management_events.is_empty() or individuals.is_empty():
        return individuals

    rates = default_event_mortality or D5_DEFAULT_EVENT_MORTALITY
    rate_table = pl.DataFrame(
        {"event": list(rates.keys()), "_default_mortality": list(rates.values())}
    )

    events = management_events
    if "mortality_probability" not in events.columns:
        events = events.with_columns(pl.lit(None, dtype=pl.Float64).alias("mortality_probability"))

    events = (
        events
        .join(rate_table, on="event", how="left")
        .with_columns(
            _event_mortality=pl.coalesce(
                [pl.col("mortality_probability"), pl.col("_default_mortality")]
            )
        )
    )

    if events.select(pl.col("_event_mortality").is_null().any()).item():
        unknown = events.filter(pl.col("_event_mortality").is_null()).get_column("event").unique().to_list()
        raise ValueError(
            "No mortality probability supplied for event(s): " + ", ".join(map(str, unknown))
        )

    invalid = events.filter((pl.col("_event_mortality") < 0) | (pl.col("_event_mortality") > 1))
    if not invalid.is_empty():
        raise ValueError("All management mortality probabilities must be in [0, 1].")

    join_keys = ["x", "y"]
    if "lifestage" in events.columns:
        if "lifestage" not in individuals.columns:
            raise ValueError("management_events contains lifestage but individuals does not.")
        join_keys.append("lifestage")

    # Collapse multiple events to one daily probability per spatial/stage key.
    daily_event_risk = (
        events
        .group_by(join_keys)
        .agg(
            (1.0 - (1.0 - pl.col("_event_mortality")).product()).alias("management_mortality_probability")
        )
    )

    return (
        individuals
        .join(daily_event_risk, on=join_keys, how="left")
        .with_columns(
            management_mortality_probability=pl.col("management_mortality_probability").fill_null(0.0),
            _management_draw=polars_random.uniform(),
        )
        .filter(pl.col("_management_draw") >= pl.col("management_mortality_probability"))
        .drop("management_mortality_probability", "_management_draw")
    )


# Daily integration point after the day's management-event map has been generated. Applying the
# event before movement makes mortality depend on the individual's location when the event occurs:
# bembidions = apply_management_mortality(bembidions, management_events_today)
# bembidions = move_to(bembidions, movement_map, 14, .8)
