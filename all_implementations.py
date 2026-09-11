import polars as pl
import polars_random
import numpy as np

n_individuals = 1000 # number of start-individuals (adults only, entering at June 8 = mid of activity phase + mean development time egg and larvae)

z = 50  # Number of years to simulate.
n_days = 365 * z

grid_size = 40  # Edge length of the square landscape in meters (1 grid cell = 1 m^2). Movement wraps around at the edges, so the population is closed and the density scale stays constant over the run.
movement_range = 5  # Maximum daily movement of adults per axis, in meters. Offsets are drawn uniformly from -movement_range to +movement_range.
start_day_offset = 158  # t=0 corresponds to day_of_year 159 (June 8), the mid-point of the window in which new adults emerge, so that age=0 is correct for the initial adult population.

dormancy_start_day = 305  # Day of year (1 = Jan 1) marking the start of winter dormancy (1 November). During dormancy, adults do not move and are not subject to density-dependent mortality, only baseline_mortality applies.
dormancy_end_day = 59  # Day of year marking the end of winter dormancy (28 February). The dormancy window wraps across the year boundary (dormancy_start_day..365, 1..dormancy_end_day), unlike the activity window.

# density dependent factors
density_dependence_radius = 3  # Epanechnikov kernel cutoff radius, in meters (1 grid cell = 1 m^2).
density_dependence_n0 = 210  # Threshold local (kernel-weighted) density at which mortality is halfway between baseline_mortality and maximum_mortality, corresponding to ~15 individuals/m^2 (14.1 m^2 effective Epanechnikov kernel area at r=3) - rough order-of-magnitude estimate, no direct literature value found.
density_dependence_k = 0.0293 # Steepness of the logistic relationship, corresponding to a transition width of about 150 (kernel-weighted) individuals between the 10% and 90% points of the mortality range. Together with density_dependence_n0 this controls the exponential left tail of the curve, which is what actually acts at the operating density - both parameters are exponentially sensitive, so use larvae_density_dependence_n_half to tune capacity instead.
baseline_mortality = 0.0015  # m_min: minimum daily mortality probability for low densitys. Applied during the active season (offset-corrected DD-mortality builds on top of this), and on its own throughout dormancy.
maximum_mortality = 0.96  # m_max: maximum daily mortality probability at very high local density, set high enough to act as an effectively enforced carrying capacity via mortality rather than an explicit cap.

maximum_fecundity = 0.25  # F_max: baseline daily fecundity (eggs/day/adult) at N(t) = 0. Sums up to a mean of 10 eggs per adult in the 40 day long activity phase.
fecundity_n_half = 30  # Local (kernel-weighted) density at which fecundity drops to half of maximum_fecundity. Lower than density_dependence_n0 so that reproduction is deferred before survival is risked: within the activity window this makes the fecundity response 7x stronger than the density-dependent mortality.
fecundity_b = np.log(2) / fecundity_n_half  # b, as used in F(t) = F_max * e^(-b * N(t)).

larvae_baseline_survival = 1.0  # S0: daily survival probability at N(t) = 0. Set to 1 since density-independent baseline mortality is now handled separately by larvae_hatching_survival_probability (avoids double-counting the same mortality source); represents purely density-dependent competition mortality.
larvae_density_dependence_n_half = 100  # N_half: number of other larvae in the same cell at which survival drops to half of S0. No reliable literature value found. This is the most usable capacity knob: the equilibrium scales nearly proportionally with it and its value does not affect stability.
larvae_density_dependence_a = 1 / larvae_density_dependence_n_half  # a = 1 / N_half, as used in S(t) = S0 / (1 + a * N(t)).


# development for different life stages
egg_development_mean = 8.5  # Mean egg development time in days, PLACEHOLDER estimate (~7-10 days) based on analogy to similarly sized carabids, no direct literature value found for B. lampros.
egg_development_sd = 1.5  # Standard deviation of individual egg development time. No literature basis - open calibration parameter.
egg_laying_survival_probability = 0.8  # Probability that a freshly laid egg survives the moment of laying (mortality through e.g. predation, infertility) before development even begins (20% immediate mortality).

larvae_development_mean = 35  # Mean larval+pupal development time in days, based on Bembidion lampros reared on Isotoma anglicana (Bilde, Axelsen & Toft 2000, Table 4: 34.93 +- 0.80 days, n=17).
larvae_development_sd = 3  # Standard deviation of individual larval development time, derived from the same source (SE=0.80, n=17 -> SD ~ 3.3, rounded).
larvae_hatching_survival_probability = 0.88  # Probability that a larva survives the moment of hatching from the egg, before density-dependent effects apply (Bilde, Axelsen & Toft 2000, Table 4: 88% survival on Isotoma anglicana diet, measured without density-dependent competition in individual rearing).

adult_max_lifespan_mean = 365  # Mean adult lifespan in days.
adult_max_lifespan_sd = 25  # Standard deviation of individual adult lifespan. No literature basis - open calibration parameter. Continues to run down during dormancy as well (age is pure calendar time).

activity_start_day = 95  # Day of year (1 = Jan 1) marking the start of the annual egg-laying activity window (~5 April), based on the ~100 degree-day-above-9C dispersal trigger reported for B. lampros.
activity_end_day = 135  # Day of year marking the end of the annual egg-laying activity window (~15 May), sized to a ~40-day reproductive period consistent with the maximum_fecundity derivation (10 lifetime eggs / ~40 days ~ 0.25 eggs/day).


# habitat quality
habitat_quality = "good"  # Global scenario switch for this run: "good", "medium" or "bad".
habitat_quality_factors_adult = {"perfect": 1.4, "good": 1.0, "medium": 0.8, "bad": 0.6}  # q_adult per habitat_quality, made up values.
habitat_quality_factors_larvae = {"perfect": 1.4, "good": 1.0, "medium": 0.8, "bad": 0.6}  # q_larvae per habitat_quality, same value as for adults has a different effect than for adults.
q_adult = habitat_quality_factors_adult[habitat_quality]
q_larvae = habitat_quality_factors_larvae[habitat_quality]
density_dependence_n0_effective = density_dependence_n0 * q_adult  # N0 scaled by habitat quality (carrying-capacity reduction).
larvae_density_dependence_n_half_effective = larvae_density_dependence_n_half * q_larvae  # N_half scaled by habitat quality (carrying-capacity reduction).
fecundity_n_half_effective = fecundity_n_half * q_adult  # fecundity_n_half scaled by the same q_adult factor as adult mortality, so habitat quality affects fecundity to the same degree.
fecundity_b_effective = np.log(2) / fecundity_n_half_effective  # b recomputed from the habitat-scaled fecundity_n_half_effective, as used in F(t) = F_max * e^(-b * N(t)).


# initial population: all individuals start as adults (simulation starts on day 159, when new adults emerge)
individuals = pl.DataFrame(
    {
        "lifestage": ["adult"] * n_individuals,
        "x": polars_random.randint(0, grid_size, size=n_individuals),
        "y": polars_random.randint(0, grid_size, size=n_individuals)
    }
)

adult_max_lifespan_draws = pl.Series(
    np.clip(np.round(np.random.normal(adult_max_lifespan_mean, adult_max_lifespan_sd, size=n_individuals)), 1, None).astype(int)
).cast(pl.Int64)

individuals = individuals.with_columns(
    age=pl.lit(0).cast(pl.Int64),
    development_time=pl.lit(0).cast(pl.Int64),
    max_lifespan=adult_max_lifespan_draws
)


# local density: Epanechnikov-kernel weighted neighbour count via spatial binning
def compute_local_density(
        individuals: pl.DataFrame,
        radius: float
) -> pl.DataFrame:

    individuals = (
        individuals
        .with_row_index("_id")
        .with_columns(
            bin_x=(pl.col("x") / radius).floor().cast(pl.Int64),
            bin_y=(pl.col("y") / radius).floor().cast(pl.Int64)
        )
    )

    bin_offsets = pl.DataFrame(
        {
            "offset_x": [dx for dx in (-1, 0, 1) for _ in (-1, 0, 1)],
            "offset_y": [dy for _ in (-1, 0, 1) for dy in (-1, 0, 1)]
        }
    )

    local_density = (
        individuals
        .join(bin_offsets, how="cross")
        .with_columns(
            candidate_bin_x=pl.col("bin_x") + pl.col("offset_x"),
            candidate_bin_y=pl.col("bin_y") + pl.col("offset_y")
        )
        .join(
            individuals.select("_id", "x", "y", "bin_x", "bin_y"),
            left_on=("candidate_bin_x", "candidate_bin_y"),
            right_on=("bin_x", "bin_y"),
            how="inner",
            suffix="_neighbour"
        )
        .filter(pl.col("_id") != pl.col("_id_neighbour"))
        .with_columns(
            distance=(
                (pl.col("x") - pl.col("x_neighbour")) ** 2
                + (pl.col("y") - pl.col("y_neighbour")) ** 2
            ).sqrt()
        )
        .filter(pl.col("distance") <= radius)
        .with_columns(
            weight=1 - (pl.col("distance") / radius) ** 2
        )
        .group_by("_id")
        .agg(local_density=pl.col("weight").sum())
    )

    return (
        individuals
        .join(local_density, on="_id", how="left")
        .with_columns(
            local_density=pl.col("local_density").fill_null(0)
        )
        .drop("_id", "bin_x", "bin_y")
    )


# adult mortality: logistic mortality probability based on local density, offset so that m(0) = baseline_mortality
def apply_density_dependent_mortality(
        individuals_with_density: pl.DataFrame,
        baseline_mortality: float,
        maximum_mortality: float,
        threshold_density: float,
        steepness: float
) -> pl.DataFrame:

    density_dependence_offset = (maximum_mortality - baseline_mortality) / (
        1 + np.exp(steepness * threshold_density)
    )

    return (
        individuals_with_density
        .with_columns(
            mortality_probability=(
                baseline_mortality
                + (maximum_mortality - baseline_mortality)
                / (
                    1
                    + (-steepness * (
                        pl.col("local_density") - threshold_density
                    )).exp()
                )
                - density_dependence_offset
            )
        )
        .with_columns(
            survival_draw=polars_random.uniform()
        )
        .filter(
            (pl.col("survival_draw") >= pl.col("mortality_probability"))
            & (pl.col("age") < pl.col("max_lifespan"))
        )
        .drop(
            "local_density",
            "mortality_probability",
            "survival_draw"
        )
    )


# adult mortality during dormancy: baseline_mortality only, no density-dependent term, no kernel computation needed
def apply_dormancy_mortality(
        adults: pl.DataFrame,
        baseline_mortality: float
) -> pl.DataFrame:

    return (
        adults
        .with_columns(survival_draw=polars_random.uniform())
        .filter(
            (pl.col("survival_draw") >= baseline_mortality)
            & (pl.col("age") < pl.col("max_lifespan"))
        )
        .drop("survival_draw")
    )


# adult fecundity: Ricker-type egg count based on local density, one row per new egg
def apply_density_dependent_fecundity(
        individuals_with_density: pl.DataFrame,
        maximum_fecundity: float,
        b: float
) -> pl.DataFrame:

    fecundity = individuals_with_density.with_columns(
        fecundity=maximum_fecundity * (-b * pl.col("local_density")).exp()
    )

    egg_counts = np.random.poisson(fecundity["fecundity"].to_numpy())

    return (
        fecundity
        .with_columns(egg_count=pl.Series(egg_counts).cast(pl.Int64))
        .filter(pl.col("egg_count") > 0)
        .select(
            pl.col("x").repeat_by(pl.col("egg_count")).explode(empty_as_null=True),
            pl.col("y").repeat_by(pl.col("egg_count")).explode(empty_as_null=True)
        )
        .with_columns(lifestage=pl.lit("egg"))
        .select("lifestage", "x", "y")
    )


# larvae mortality: Hassell-type survival based on same-cell larval count
def apply_density_dependent_larvae_survival(
        individuals: pl.DataFrame,
        baseline_survival: float,
        n_half: float
) -> pl.DataFrame:

    a = 1 / n_half

    larvae = individuals.filter(pl.col("lifestage") == pl.lit("larvae"))
    other_lifestages = individuals.filter(pl.col("lifestage") != pl.lit("larvae"))

    surviving_larvae = (
        larvae
        .with_columns(cell_count=pl.len().over("x", "y"))
        .with_columns(local_density=pl.col("cell_count") - 1)
        .with_columns(
            survival_probability=baseline_survival / (1 + a * pl.col("local_density"))
        )
        .with_columns(survival_draw=polars_random.uniform())
        .filter(pl.col("survival_draw") < pl.col("survival_probability"))
        .drop("cell_count", "local_density", "survival_probability", "survival_draw")
    )

    return pl.concat([surviving_larvae, other_lifestages])


# egg hatching: age-based transition to larvae, once development_time is reached, with immediate hatching mortality
def apply_egg_hatching(
        individuals: pl.DataFrame,
        development_time_mean: float,
        development_time_sd: float,
        hatching_survival_probability: float
) -> pl.DataFrame:

    individuals = individuals.with_columns(
        hatching=(pl.col("lifestage") == pl.lit("egg")) & (pl.col("age") >= pl.col("development_time"))
    )

    new_development_times = pl.Series(
        np.clip(
            np.round(np.random.normal(development_time_mean, development_time_sd, size=individuals.height)),
            1,
            None
        ).astype(int)
    ).cast(pl.Int64)

    individuals = individuals.with_columns(
        lifestage=pl.when(pl.col("hatching")).then(pl.lit("larvae")).otherwise(pl.col("lifestage")),
        age=pl.when(pl.col("hatching")).then(0).otherwise(pl.col("age")).cast(pl.Int64),
        development_time=pl.when(pl.col("hatching")).then(new_development_times).otherwise(pl.col("development_time")).cast(pl.Int64),
        hatching_survival_draw=polars_random.uniform()
    )

    return (
        individuals
        .filter((~pl.col("hatching")) | (pl.col("hatching_survival_draw") <= hatching_survival_probability))
        .drop("hatching", "hatching_survival_draw")
    )


# larvae hatching: age-based transition to adult, once development_time is reached
def apply_larvae_hatching(
        individuals: pl.DataFrame,
        max_lifespan_mean: float,
        max_lifespan_sd: float
) -> pl.DataFrame:

    hatching = (pl.col("lifestage") == pl.lit("larvae")) & (pl.col("age") >= pl.col("development_time"))

    new_max_lifespans = pl.Series(
        np.clip(
            np.round(np.random.normal(max_lifespan_mean, max_lifespan_sd, size=individuals.height)),
            1,
            None
        ).astype(int)
    ).cast(pl.Int64)

    return individuals.with_columns(
        lifestage=pl.when(hatching).then(pl.lit("adult")).otherwise(pl.col("lifestage")),
        age=pl.when(hatching).then(0).otherwise(pl.col("age")).cast(pl.Int64),
        max_lifespan=pl.when(hatching).then(new_max_lifespans).otherwise(pl.col("max_lifespan")).cast(pl.Int64)
    )


egg_population = []
larvae_population = []
adult_population = []

# main simulation loop: one iteration per day, z years total
for t in range(n_days):

    day_of_year = ((t + start_day_offset) % 365) + 1
    in_dormancy = day_of_year >= dormancy_start_day or day_of_year <= dormancy_end_day

    # movement and wrapping are skipped entirely during dormancy (adults do not move)
    if not in_dormancy:
        individuals = (
            individuals
            .with_columns(
                x=pl
                .when(lifestage=pl.lit("adult"))
                .then(
                    polars_random.randint(
                        pl.col("x") - movement_range,
                        pl.col("x") + movement_range + 1
                    )
                )
                .otherwise("x"),
                y=pl
                .when(lifestage=pl.lit("adult"))
                .then(
                    polars_random.randint(
                        pl.col("y") - movement_range,
                        pl.col("y") + movement_range + 1
                    )
                )
                .otherwise("y")
            )
        )

        # wrapping: individuals leaving one edge of the landscape re-enter on the opposite edge
        individuals = individuals.with_columns(
            x=(pl.col("x") % grid_size + grid_size) % grid_size,
            y=(pl.col("y") % grid_size + grid_size) % grid_size
        )

    individuals = individuals.with_columns(age=pl.col("age") + 1)

    adults = individuals.filter(pl.col("lifestage") == pl.lit("adult"))
    other_lifestages = individuals.filter(pl.col("lifestage") != pl.lit("adult"))

    # egg laying and density-dependent adult mortality are both skipped during dormancy;
    # only baseline_mortality applies, and no kernel density computation is needed at all
    if in_dormancy:
        laid_eggs = adults.clear().select("x", "y").with_columns(lifestage=pl.lit("egg"))
        surviving_adults = apply_dormancy_mortality(adults, baseline_mortality=baseline_mortality)
    else:
        adults_with_density = compute_local_density(adults, radius=density_dependence_radius)

        # egg laying
        if activity_start_day <= day_of_year <= activity_end_day:
            laid_eggs = apply_density_dependent_fecundity(
                adults_with_density,
                maximum_fecundity=maximum_fecundity,
                b=fecundity_b_effective
            )
        else:
            laid_eggs = adults_with_density.clear().select("x", "y").with_columns(lifestage=pl.lit("egg"))

        # density-dependent adult mortality
        surviving_adults = apply_density_dependent_mortality(
            adults_with_density,
            baseline_mortality=baseline_mortality,
            maximum_mortality=maximum_mortality,
            threshold_density=density_dependence_n0_effective,
            steepness=density_dependence_k
        )

    # immediate egg mortality at the moment of laying
    laid_eggs = laid_eggs.filter(polars_random.uniform() <= egg_laying_survival_probability)

    laid_eggs = (
        laid_eggs
        .with_columns(
            age=pl.lit(0).cast(pl.Int64),
            development_time=pl.Series(
                np.clip(
                    np.round(np.random.normal(egg_development_mean, egg_development_sd, size=laid_eggs.height)),
                    1,
                    None
                ).astype(int)
            ).cast(pl.Int64),
            max_lifespan=pl.lit(0).cast(pl.Int64)
        )
        .select("lifestage", "x", "y", "age", "development_time", "max_lifespan")
    )

    # egg hatching
    other_lifestages = apply_egg_hatching(
        other_lifestages,
        development_time_mean=larvae_development_mean,
        development_time_sd=larvae_development_sd,
        hatching_survival_probability=larvae_hatching_survival_probability
    )

    # larvae mortality
    other_lifestages = apply_density_dependent_larvae_survival(
        other_lifestages,
        baseline_survival=larvae_baseline_survival,
        n_half=larvae_density_dependence_n_half_effective
    )

    # larvae hatching
    other_lifestages = apply_larvae_hatching(
        other_lifestages,
        max_lifespan_mean=adult_max_lifespan_mean,
        max_lifespan_sd=adult_max_lifespan_sd
    )
    newly_adult = other_lifestages.filter(pl.col("lifestage") == pl.lit("adult"))
    other_lifestages = other_lifestages.filter(pl.col("lifestage") != pl.lit("adult"))

    individuals = pl.concat([surviving_adults, other_lifestages, laid_eggs, newly_adult])

    egg_population.append(individuals.filter(pl.col("lifestage") == pl.lit("egg")).height)
    larvae_population.append(individuals.filter(pl.col("lifestage") == pl.lit("larvae")).height)
    adult_population.append(individuals.filter(pl.col("lifestage") == pl.lit("adult")).height)

    print(t)


# plot population trajectories per lifestage
import matplotlib.pyplot as plt

plt.figure()
plt.plot(range(n_days), egg_population)
plt.xlabel("Time (days)")
plt.ylabel("Number of eggs")
plt.show()

plt.figure()
plt.plot(range(n_days), larvae_population)
plt.xlabel("Time (days)")
plt.ylabel("Number of larvae")
plt.show()

plt.figure()
plt.plot(range(n_days), adult_population)
plt.xlabel("Time (days)")
plt.ylabel("Number of adults")
plt.show()