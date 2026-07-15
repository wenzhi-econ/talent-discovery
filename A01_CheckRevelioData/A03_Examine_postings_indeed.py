"""
Task:
(1) Inspect the Revelio ``postings_indeed`` table in Microsoft Fabric.

Workflow:
(1) Read the table and retain the validated columns of interest.
(2) Specify the unit of observation, identifiers, and variable meanings.
(3) Print the dataset documentation using the live Spark schema.
(4) Export distinct combinations of the selected standardized role variables.
(5) Export a lightweight sample of 1,000 observations.

The exported value files describe values observed when this script is run.
They should not be interpreted as a permanent or theoretical Revelio taxonomy.

Wang Wenzhi, with the help of Codex
Time: 2026-07-15
"""

from pyspark.sql import SparkSession

"""
Notes:
(1) There is no need to add extension to the file name.
(2) Save the desired output as a Delta table and a Parquet dataset with the ``write`` method.
"""
DATASET_NAME = "postings_indeed"
OUTPUT_TABLE_NAME = "z_universe_roles_indeed"
OUTPUT_ROLE_UNIVERSE = "Files/WenzhiW/List_StandardizedRoles_InIndeedPostings"
SAMPLE_SIZE = 1000
OUTPUT_SAMPLE_TABLE = "z_sample_postings_indeed"
OUTPUT_SAMPLE = "Files/WenzhiW/Sample_IndeedPostings"


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Read the Indeed posting data and select validated columns
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


spark = SparkSession.builder.getOrCreate()
postings_indeed = spark.read.table(DATASET_NAME)
postings_indeed = postings_indeed.select(
    "job_id",
    "rcid",
    "company",
    "rics_k50",
    "rics_k200",
    "rics_k400",
    "title_raw",
    "title_translated",
    "job_category",
    "role_k50",
    "role_k150",
    "role_k300",
    "role_k500",
    "role_k1000",
    "role_k1250",
    "role_k1500",
    "location_raw",
    "region",
    "country",
    "state",
    "metro_area",
    "salary",
    "post_date",
    "remove_date",
    "ultimate_parent_rcid",
    "ultimate_parent_company_name",
    "onet_code",
    "onet_title",
    "remote_type",
    "jobtitle",
    "description",
    "salary_min",
    "salary_max",
    "salary_predicted",
)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Specify the dataset and variable documentation
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


DATASET_SPEC = {
    "unit_of_observation": "One individual job posting collected from Indeed",
    "candidate_key": ("job_id",),
    "linkage_identifiers": (
        "job_id",
        "rcid",
        "ultimate_parent_rcid",
    ),
}
"""
Notes:
(1) "job_id" is Revelio's documented posting key.
(2) To keep this inspection lightweight, the script records "job_id" as the candidate key but
    does not run a computational uniqueness check.
(3) Variable meanings follow the delivery PDF and https://www.data-dictionary.reveliolabs.com/data.html.
    The delivery PDF is used for delivered fields omitted from the current public dictionary.
(4) ``region`` and ``jobtitle`` are absent from both reference dictionaries and are explicitly
    identified as delivery-specific.
"""

VARIABLE_MEANINGS = {
    "job_id": "Posting key.",
    "rcid": "Revelio Labs company ID.",
    "company": "Company name.",
    "rics_k50": "Revelio Labs employer-industry classification with 50 discrete categories.",
    "rics_k200": "Revelio Labs employer-industry classification with 200 discrete categories.",
    "rics_k400": "Revelio Labs employer-industry classification with 400 discrete categories.",
    "title_raw": "Posting job title as reported in the raw posting.",
    "title_translated": "Raw posting job title translated into English.",
    "job_category": "Mapped posting role with 7 discrete levels.",
    "role_k50": "Aggregated posting role with 50 discrete levels.",
    "role_k150": "Aggregated posting role with 150 discrete levels.",
    "role_k300": "Aggregated posting role with 300 discrete levels.",
    "role_k500": "Aggregated posting role with 500 discrete levels.",
    "role_k1000": "Aggregated posting role with 1,000 discrete levels.",
    "role_k1250": "Aggregated posting role with 1,250 discrete levels.",
    "role_k1500": "Aggregated posting role with 1,500 discrete levels.",
    "location_raw": "Raw location of the job posting.",
    "region": "Delivery-specific posting-region field not defined in either reference dictionary.",
    "country": "Country of the job posting, imputed from the raw location.",
    "state": "State of the job posting, imputed from the raw location.",
    "metro_area": "Metropolitan area of the job posting, imputed from the raw location.",
    "salary": (
        "Annualized salary in USD, taken from the posting when available and otherwise estimated "
        "by Revelio's salary model."
    ),
    "post_date": "Date on which the job was posted.",
    "remove_date": "Date on which the posting was removed; may be null.",
    "ultimate_parent_rcid": "Revelio Labs company ID for the parent company.",
    "ultimate_parent_company_name": "Name of the ultimate parent company.",
    "onet_code": "Eight-digit predicted O*NET code of the posted job, mapped at the role_k1500 level.",
    "onet_title": "O*NET title of the posted job, mapped at the role_k1500 level.",
    "remote_type": "Type of remote work offered; unspecified postings are categorized as fully in office.",
    "jobtitle": "Delivery-specific job-title field not defined in either reference dictionary.",
    "description": "Raw full-text description from the job posting.",
    "salary_min": "Minimum of the salary range provided in the posting, in USD; null when salary is predicted.",
    "salary_max": "Maximum of the salary range provided in the posting, in USD; null when salary is predicted.",
    "salary_predicted": (
        "Boolean indicator for whether the posting salary was predicted using Revelio's salary model."
    ),
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
for field in postings_indeed.schema.fields:
    print(
        f"  - {field.name}\n"
        f"    - Label: {VARIABLE_MEANINGS[field.name]}\n"
        f"    - Spark schema: [{field.dataType.simpleString()}; nullable={field.nullable}]\n"
    )


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 4. Export observed universe of standardized roles
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


ROLE_VARIABLES = (
    "job_category",
    "role_k50",
    "role_k150",
    "role_k300",
    "role_k500",
    "role_k1000",
    "role_k1250",
    "role_k1500",
)

"""
Notes:
(1) Keep only the standardized role variables and remove repeated combinations.
(2) Keep all 8 variables to preserve the observed mapping across role-taxonomy levels.
"""

role_universe = postings_indeed.select(*ROLE_VARIABLES).dropDuplicates()

(
    role_universe.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(OUTPUT_TABLE_NAME)
)
print(f"Table saved: {OUTPUT_TABLE_NAME}.")

(
    spark.read.table(OUTPUT_TABLE_NAME)
    .coalesce(1)
    .write.mode("overwrite")
    .parquet(OUTPUT_ROLE_UNIVERSE)
)
print(f"Parquet dataset saved: {OUTPUT_ROLE_UNIVERSE}.")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 5. Export a lightweight sample of observations
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


sample_data = postings_indeed.limit(SAMPLE_SIZE)

(
    sample_data.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(OUTPUT_SAMPLE_TABLE)
)
print(f"Table saved: {OUTPUT_SAMPLE_TABLE}.")

(spark.read.table(OUTPUT_SAMPLE_TABLE).coalesce(1).write.mode("overwrite").parquet(OUTPUT_SAMPLE))
print(f"Parquet dataset saved: {OUTPUT_SAMPLE}.")
