"""
Task:
(1) Inspect the Revelio ``user_demography`` table in Microsoft Fabric.

Workflow:
(1) Read the table and retain the validated columns of interest.
(2) Specify the unit of observation, identifiers, and variable meanings.
(3) Print the dataset documentation using the live Spark schema.
(4) Export a random sample of 1,000 observations.

Wang Wenzhi, with the help of Codex
Time: 2026-07-15
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

"""
Notes:
(1) There is no need to add extension to the file name.
(2) Save the desired output as a Delta table and a Parquet dataset with the ``write`` method.
"""
DATASET_NAME = "user_demography"
SAMPLE_SIZE = 1000
RANDOM_SEED = 20260715
OUTPUT_SAMPLE_TABLE = "z_sample_user_demography"
OUTPUT_SAMPLE = "Files/WenzhiW/Sample_UserDemography"


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Read the demographic data and select validated columns
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


spark = SparkSession.builder.getOrCreate()
user_demography = spark.read.table(DATASET_NAME)
user_demography = user_demography.select(
    "user_id",
    "firstname",
    "lastname",
    "fullname",
    "f_prob",
    "m_prob",
    "white_prob",
    "black_prob",
    "api_prob",
    "hispanic_prob",
    "native_prob",
    "multiple_prob",
    "prestige",
    "highest_degree",
    "sex_predicted",
    "ethnicity_predicted",
    "user_location",
    "user_country",
    "profile_title",
    "updated_dt",
    "numconnections",
    "profile_summary",
)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Specify the dataset and variable documentation
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


DATASET_SPEC = {
    "unit_of_observation": "One individual public profile",
    "candidate_key": ("user_id",),
    "linkage_identifiers": ("user_id",),
}
"""
Notes:
(1) "user_id" is Revelio's documented individual identifier.
(2) To keep this inspection lightweight, the script records "user_id" as the candidate key but
    does not run a computational uniqueness check.
(3) The delivered ``user_demography`` table corresponds to fields documented under Revelio's
    individual-level User File.
"""

VARIABLE_MEANINGS = {
    "user_id": "Revelio Labs identifier for an individual public profile.",
    "firstname": "First name parsed from the user's full name.",
    "lastname": "Last name parsed from the user's full name.",
    "fullname": "Full name as reported in the online public profile.",
    "f_prob": "Probability of the user being female.",
    "m_prob": "Probability of the user being male.",
    "white_prob": "Probability of the user being Non-Hispanic White.",
    "black_prob": "Probability of the user being Black or African American.",
    "api_prob": "Probability of the user being Asian or Pacific Islander.",
    "hispanic_prob": "Probability of the user being Hispanic or Latino.",
    "native_prob": "Probability of the user being American Indian or Alaskan Native.",
    "multiple_prob": "Probability of the user being two or more races.",
    "prestige": "User prestige predicted by Revelio's prestige model.",
    "highest_degree": "Highest level of education reported for the user.",
    "sex_predicted": "Predicted sex of the user.",
    "ethnicity_predicted": "Predicted ethnicity of the user.",
    "user_location": "User location as reported in the online public profile.",
    "user_country": "Country of the user imputed from the reported location.",
    "profile_title": "Profile title as reported in the online public profile.",
    "updated_dt": "Date on which the user's profile was last updated in Revelio's data.",
    "numconnections": "Number of connections on the user's profile, capped at 500.",
    "profile_summary": "Short profile summary written by the user.",
}


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 3. Print the dataset documentation
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


print("\n" + "=" * 100)
print(f"Dataset name: {DATASET_NAME}  ")
print(f"Unit of observation: {DATASET_SPEC['unit_of_observation']}  ")
print(f"Candidate key: {' + '.join(DATASET_SPEC['candidate_key'])}  ")
print(
    f"Potential identifiers used to link with other datasets: {', '.join(DATASET_SPEC['linkage_identifiers'])}  "
)
print("Variable label and spark schema:  ")
for field in user_demography.schema.fields:
    print(
        f"  - {field.name}\n"
        f"    - Label: {VARIABLE_MEANINGS[field.name]}\n"
        f"    - Spark schema: [{field.dataType.simpleString()}; nullable={field.nullable}]\n"
    )


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 4. Export a random sample of observations
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


sample_data = user_demography.orderBy(F.rand(RANDOM_SEED)).limit(SAMPLE_SIZE)

(
    sample_data.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(OUTPUT_SAMPLE_TABLE)
)
print(f"Table saved: {OUTPUT_SAMPLE_TABLE}.")

(spark.read.table(OUTPUT_SAMPLE_TABLE).coalesce(1).write.mode("overwrite").parquet(OUTPUT_SAMPLE))
print(f"Parquet dataset saved: {OUTPUT_SAMPLE}.")
