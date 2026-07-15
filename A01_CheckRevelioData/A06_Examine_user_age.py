"""
Task:
(1) Inspect the Revelio ``user_age`` table in Microsoft Fabric.

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
DATASET_NAME = "user_age"
SAMPLE_SIZE = 1000
RANDOM_SEED = 20260715
OUTPUT_SAMPLE_TABLE = "z_sample_user_age"
OUTPUT_SAMPLE = "Files/WenzhiW/Sample_UserAge"


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Read the age data and select validated columns
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


spark = SparkSession.builder.getOrCreate()
user_age = spark.read.table(DATASET_NAME)
user_age = user_age.select(
    "user_id",
    "earliest_grad_year",
    "estimated_birth_year",
    "age",
)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Specify the dataset and variable documentation
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


DATASET_SPEC = {
    "unit_of_observation": "One age-estimation record for one individual public profile",
    "candidate_key": ("user_id",),
    "linkage_identifiers": ("user_id",),
}
"""
Notes:
(1) "user_id" is Revelio's documented individual identifier.
(2) To keep this inspection lightweight, the script records "user_id" as the candidate key but
    does not run a computational uniqueness check.
(3) The delivery PDF and public data dictionary do not document the ``user_age`` table. The labels
    below therefore describe the delivered field names and should be checked against the sample
    before the variables are used substantively.
"""

VARIABLE_MEANINGS = {
    "user_id": "Revelio Labs identifier for an individual public profile.",
    "earliest_grad_year": "Earliest graduation year recorded for the user in the delivered age table.",
    "estimated_birth_year": "Estimated year of birth supplied in the delivered age table.",
    "age": "Estimated age supplied in the delivered age table; the reference date is not documented.",
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
for field in user_age.schema.fields:
    print(
        f"  - {field.name}\n"
        f"    - Label: {VARIABLE_MEANINGS[field.name]}\n"
        f"    - Spark schema: [{field.dataType.simpleString()}; nullable={field.nullable}]\n"
    )


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 4. Export a random sample of observations
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


sample_data = user_age.orderBy(F.rand(RANDOM_SEED)).limit(SAMPLE_SIZE)

(
    sample_data.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(OUTPUT_SAMPLE_TABLE)
)
print(f"Table saved: {OUTPUT_SAMPLE_TABLE}.")

(spark.read.table(OUTPUT_SAMPLE_TABLE).coalesce(1).write.mode("overwrite").parquet(OUTPUT_SAMPLE))
print(f"Parquet dataset saved: {OUTPUT_SAMPLE}.")
