# Kaylee Faherty
# Maine Town Penetration Analysis Code - DS5110 Project

import pandas as pd
import matplotlib.pyplot as plt

# --- Load subscriber data ---
subs = pd.read_csv("combined_cleaned_subscriber_data.csv", low_memory=False)

# --- Load Maine population data (headers are in 3rd row) ---
pop = pd.read_excel("Maine_Cities_Towns_Population_Estimates_2024.xlsx", header=2)

# --- Standardize column names ---
pop.columns = pop.columns.str.strip().str.lower().str.replace(" ", "_")

# --- Select town + population columns ---
pop = pop[['town', '2024_population']].rename(columns={'2024_population':'population'})

# --- Normalize town names in population data ---
# Strip only 'city' and 'town' suffixes, keep 'plantation' and 'UT' distinct
pop['town'] = pop['town'].str.replace(r'\s+(city|town)$', '', regex=True).str.strip()

# --- Normalize subscriber city names ---
subs['city'] = subs['city'].str.strip().str.title()

# --- Manual mapping for ambiguous cases ---
manual_map = {
    'Lincoln': 'Lincoln town',
    'Rangeley': 'Rangeley town',
    'Unity': 'Unity town',
    # Add more if anomalies appear
}
subs['city_normalized'] = subs['city'].map(manual_map).fillna(subs['city'])

# --- Aggregate subscribers by normalized city ---
subs_by_city = subs.groupby('city_normalized').size().reset_index(name='subscribers')
subs_by_city = subs_by_city.rename(columns={'city_normalized':'town'})

# --- Merge with population data ---
merged = subs_by_city.merge(pop, on='town', how='inner')

# --- Calculate penetration percentage ---
merged['penetration_pct'] = (merged['subscribers'] / merged['population']) * 100

# --- Diagnostic: flag anomalies >100% ---
anomalies = merged[merged['penetration_pct'] > 100]
print("Anomalies (penetration > 100%):")
print(anomalies[['town','subscribers','population','penetration_pct']])

# --- Exclude anomalies from graphs ---
merged_clean = merged[merged['penetration_pct'] <= 100]

# --- Top 10 towns by penetration ---
top10 = merged_clean.sort_values('penetration_pct', ascending=False).head(10)
plt.figure(figsize=(10,6))
plt.barh(top10['town'], top10['penetration_pct'], color='steelblue')
plt.xlabel("Penetration (%)")
plt.title("Top 10 Maine Towns by Subscriber Penetration (Excluding >100%)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# --- Bottom 10 towns by penetration ---
bottom10 = merged_clean.sort_values('penetration_pct', ascending=True).head(10)
plt.figure(figsize=(10,6))
plt.barh(bottom10['town'], bottom10['penetration_pct'], color='orange')
plt.xlabel("Penetration (%)")
plt.title("Bottom 10 Maine Towns by Subscriber Penetration (Excluding >100%)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# --- Histogram of penetration rates ---
plt.figure(figsize=(10,6))
plt.hist(merged_clean['penetration_pct'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel("Penetration (%)")
plt.ylabel("Number of Towns")
plt.title("Distribution of Subscriber Penetration Across Maine Towns (Excluding >100%)")
plt.tight_layout()
plt.show()


