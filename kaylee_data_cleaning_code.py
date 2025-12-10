#Kaylee Faherty
#Data File Cleaning & Compiling Code - DS5110 Project

import pandas as pd
import os
import re

# --- Cleaning Function ---
def clean_subscriber_data(file_path):
    """
    Clean a single subscriber CSV file.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Cleaned subscriber data.
    """
    df = pd.read_csv(file_path, low_memory=False)
    
    #Column Cleanup
    # Drop unused columns
    irrelevant_cols = ['Dist ID', 'Route ID', 'Legacy Acct ID']
    df.drop(columns=[col for col in irrelevant_cols if col in df.columns], inplace=True)

    # Standardize column names: lowercase, underscores
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    #Identifier Cleanup
    # Pad zip codes to 5 digits with leading 0
    if 'zip_code' in df.columns:
        df['zip_code'] = df['zip_code'].astype(str).str.zfill(5)

    # Drop rows missing critical identifiers
    critical_cols = [col for col in ['subscriber_id', 'publication'] if col in df.columns]
    if critical_cols:
        df.dropna(subset=critical_cols, inplace=True)

    # Map publication codes
    pub_map = {
        'MTM_PT': 'Portland Press Herald',
        'MTM_KJ': 'Kennebec Journal',
        'MTM_MS': 'Morning Sentinel',
        'AMG_TR': 'Times Record',
        'SMG_SJ': 'Sun Journal',
        'SMG_AD': 'Advertisor Democrat',
        'SMG_BC': 'Bethel Citizen',
        'SMG_FJ': 'Franklin Journal',
        'SMG_LFA': 'Livermore Falls Advertisor',
        'SMG_RFT': 'Rumford Falls Times',
        'SMG_RH': 'Rangeley Highlander'
    }
    if 'publication' in df.columns:
        df['publication_name'] = df['publication'].map(pub_map)

    # Convert date columns
    for col in ['start_date', 'end_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    if 'laststartdate' in df.columns and 'file_month' in df.columns and 'file_year' in df.columns:
        df['laststartdate'] = pd.to_datetime(df['laststartdate'], errors='coerce')
        snapshot_date = pd.to_datetime(
            df['file_month'].astype(str) + " " + df['file_year'].astype(str),
            format="%B %Y",   # Month name + Year
            errors='coerce'
        )
        df['months_since_last_start'] = (
            (snapshot_date.dt.to_period("M") - df['laststartdate'].dt.to_period("M"))
            .apply(lambda x: x.n if pd.notnull(x) else None)
        )


    # Digital flag
    if 'day_pattern' in df.columns:
        df['is_digital'] = df['day_pattern'].str.upper().eq('O7DAY')

    # Month and year
    if 'start_date' in df.columns:
        df['start_month'] = df['start_date'].dt.month
        df['start_year'] = df['start_date'].dt.year

    # Fill missing categorical values with "Unknown"
    for col in ['town', 'publication_name']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    # Fill missing numeric values with median
    num_cols = df.select_dtypes(include='number').columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Drop duplicate subscriber records
    if 'subscriber_id' in df.columns:
        df.drop_duplicates(subset=['subscriber_id'], inplace=True)

    return df


# --- Batch Processing Loop ---
folder_path = "."  # current folder
cleaned_dfs = []

for filename in os.listdir(folder_path):
    if filename.endswith(".csv") and filename != "combined_cleaned_subscriber_data.csv":   
        file_path = os.path.join(folder_path, filename)
        try:
            cleaned_df = clean_subscriber_data(file_path)
            cleaned_df['source_file'] = filename

            # --- Add month and year from filename ---
            match = re.search(r'(\d{1,2})\.(\d{1,2})\.(\d{2,4})', filename)
            if match:
                month_num = int(match.group(1))  # first number = month
                year_str = match.group(3)
                # Keep month as full name string (e.g., "October")
                cleaned_df['file_month'] = pd.to_datetime(str(month_num), format='%m').strftime('%B')
                cleaned_df['file_year'] = "20" + year_str if len(year_str) == 2 else year_str
            else:
                cleaned_df['file_month'] = "Unknown"
                cleaned_df['file_year'] = "Unknown"

            # --- Build snapshot_date from month/year strings ---
            snapshot_date = pd.to_datetime(
                cleaned_df['file_month'].astype(str) + " " + cleaned_df['file_year'].astype(str),
                format="%B %Y",   # Month name + Year
                errors='coerce'
            )


            # --- Months since original and current subscription start ---
            if 'originalstartdate' in cleaned_df.columns:
                cleaned_df['originalstartdate'] = pd.to_datetime(cleaned_df['originalstartdate'], errors='coerce')
                cleaned_df['months_since_original_start'] = (
                    (snapshot_date.dt.to_period("M") - cleaned_df['originalstartdate'].dt.to_period("M"))
                    .apply(lambda x: x.n if pd.notnull(x) else None)
                )

            if 'laststartdate' in cleaned_df.columns:
                cleaned_df['laststartdate'] = pd.to_datetime(cleaned_df['laststartdate'], errors='coerce')
                cleaned_df['months_since_last_start'] = (
                    (snapshot_date.dt.to_period("M") - cleaned_df['laststartdate'].dt.to_period("M"))
                    .apply(lambda x: x.n if pd.notnull(x) else None)
                )

            cleaned_dfs.append(cleaned_df)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Combine all cleaned data
# Only keep non-empty and non-all-NA dataframes
non_empty_dfs = [df for df in cleaned_dfs if not df.empty and not df.isna().all().all()]

if non_empty_dfs:
    all_data = pd.concat(non_empty_dfs, ignore_index=True)
    all_data.to_csv("combined_cleaned_subscriber_data.csv", index=False)
    print(f"Combined {len(non_empty_dfs)} files into {len(all_data)} rows.")
else:
    print("No valid dataframes to combine.")

