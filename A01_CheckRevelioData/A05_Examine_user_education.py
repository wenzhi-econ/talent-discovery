"""
Task:
(1) Inspect the Revelio ``user_education`` table in Microsoft Fabric.

Workflow:
(1) Read the table and retain the validated columns of interest.
(2) Specify the unit of observation, identifiers, and variable meanings.
(3) Print the dataset documentation using the live Spark schema.
(4) Export the distinct combinations of the standardized degree and field variables.
(5) Export a lightweight sample of 1,000 observations.

The exported value files describe values observed when this script is run.
They should not be interpreted as a permanent or theoretical Revelio taxonomy.

Wang Wenzhi, with the help of Codex
Time: 2026-07-14
"""

from pyspark.sql import SparkSession

"""
Notes:
(1) There is no need to add extension to the file name.
(2) Save the desired output as a Delta table and a Parquet dataset with the ``write`` method.
"""
DATASET_NAME = "user_education"
OUTPUT_TABLE_NAME = "z_universe_education"
OUTPUT_EDUCATION_UNIVERSE = "Files/WenzhiW/List_DegreeFields_InUserEducation"
SAMPLE_SIZE = 1000
OUTPUT_SAMPLE_TABLE = "z_sample_user_education"
OUTPUT_SAMPLE = "Files/WenzhiW/Sample_UserEducation"


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Read the education data and select validated columns
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


spark = SparkSession.builder.getOrCreate()
user_education = spark.read.table(DATASET_NAME)
user_education = user_education.select(
    "user_id",
    "university_raw",
    "university_name",
    "rsid",
    "education_number",
    "startdate",
    "enddate",
    "degree",
    "field",
    "degree_raw",
    "field_raw",
    "university_country",
    "university_location",
    "ultimate_parent_school_name",
    "ultimate_parent_rsid",
    "description",
)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Specify the dataset and variable documentation
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


DATASET_SPEC = {
    "unit_of_observation": "One educational record reported by one individual",
    "candidate_key": ("user_id", "education_number"),
    "linkage_identifiers": (
        "user_id",
        "rsid",
        "ultimate_parent_rsid",
    ),
}
"""
Notes:
(1) "user_id" is Revelio's documented individual identifier, but the data dictionary does not
    document a standalone identifier for an education record.
(2) The script records "user_id" and "education_number" as the candidate composite key but does
    not run a computational uniqueness check.
"""

VARIABLE_MEANINGS = {
    "user_id": "Revelio Labs identifier for an individual public profile.",
    "university_raw": "School name as reported in the public profile.",
    "university_name": "School name after Revelio Labs mapping.",
    "rsid": "Revelio Labs identifier for the mapped school.",
    "education_number": "Chronological order of the education record within the user profile.",
    "startdate": "Start date of the education record.",
    "enddate": "End date of the education record.",
    "degree": "Degree title after Revelio Labs mapping.",
    "field": "Degree field after Revelio Labs mapping.",
    "degree_raw": "Degree title as reported in the public profile.",
    "field_raw": "Field of study as reported in the public profile.",
    "university_country": "Country of the school after Revelio Labs mapping.",
    "university_location": "Location of the school after Revelio Labs mapping.",
    "ultimate_parent_school_name": "Name of the ultimate parent school or institution.",
    "ultimate_parent_rsid": "Revelio identifier for the ultimate parent school.",
    "description": "Raw education description entered in the public profile.",
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
for field in user_education.schema.fields:
    print(
        f"  - {field.name}\n"
        f"    - Label: {VARIABLE_MEANINGS[field.name]}\n"
        f"    - Spark schema: [{field.dataType.simpleString()}; nullable={field.nullable}]\n"
    )


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 4. Export observed universe of standardized education categories
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


"""
Notes:
(1) In the first round of data inspection, I decide to not get the universe of degrees and fields.
"""

# EDUCATION_VARIABLES = (
#     "degree",
#     "field",
# )

# """
# Notes:
# (1) Keep only Revelio's standardized degree and field variables and remove repeated combinations.
# (2) Keep both variables to preserve the observed mapping between degrees and fields of study.
# """

# education_universe = user_education.select(*EDUCATION_VARIABLES).dropDuplicates()

# (
#     education_universe.write.format("delta")
#     .mode("overwrite")
#     .option("overwriteSchema", "true")
#     .saveAsTable(OUTPUT_TABLE_NAME)
# )
# print(f"Table saved: {OUTPUT_TABLE_NAME}.")

# (
#     spark.read.table(OUTPUT_TABLE_NAME)
#     .coalesce(1)
#     .write.mode("overwrite")
#     .parquet(OUTPUT_EDUCATION_UNIVERSE)
# )
# print(f"Parquet dataset saved: {OUTPUT_EDUCATION_UNIVERSE}.")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 5. Export a lightweight sample of observations
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


sample_data = user_education.limit(SAMPLE_SIZE)

(
    sample_data.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(OUTPUT_SAMPLE_TABLE)
)
print(f"Table saved: {OUTPUT_SAMPLE_TABLE}.")

(spark.read.table(OUTPUT_SAMPLE_TABLE).coalesce(1).write.mode("overwrite").parquet(OUTPUT_SAMPLE))
print(f"Parquet dataset saved: {OUTPUT_SAMPLE}.")
