import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print('Enter the name of the sample you would like to analyse:')
PDX_sample = input()

USE_CONTROL = 'Doxorubicin'  # Empirically use 'AraC' or 'Doxorubicin' based on highest sensitivity

############################################
### Import raw absolute counts for PDX sample
############################################
try:
    raw_df = pd.read_csv(PDX_sample + '/raw_data/' + PDX_sample + '_absolute_counts_collated.csv')
except:
    print('Please ensure the sample ID (' + PDX_sample +  ') is correct and that a directory exists for that sample.')
    print('Please re-enter a valid sample ID:')
    PDX_sample = input()
    raw_df = pd.read_csv(PDX_sample + '/raw_data/' + PDX_sample + '_absolute_counts_collated.csv')

# If Image_Metadata_Well in single digit format, correct (e.g. A1 to A01)
raw_df['Row'] = raw_df['Image_Metadata_Well'].str.extract(r'(^\w)')
raw_df['Col'] = raw_df['Image_Metadata_Well'].str.extract(r'^\w(\d{1,2})')
raw_df['Col'] = raw_df['Col'].str.zfill(2)
raw_df['Image_Metadata_Well'] = raw_df['Row'].map(str) + raw_df['Col'].map(str)

print('Exporting QC heatmaps for ' + PDX_sample + '.')
# Export plate heatmaps for visual QC checking
def plate_heatmap(plate, id=0):
    """
    Reshape given raw data file into a 384-well array and plot to heatmap.
    :param file: pandas DataFrame name
    :return: returns seaborn heatmap as png
    """

    global xticks, yticks
    plate = plate.iloc[:, 0]

    ## Create directory in figures to export heatmaps
    Path(PDX_sample + '/figures/QC/heatmaps').mkdir(parents=True, exist_ok=True)

    if plate.shape[0] == 384:
        plate_reshape = (16, 24)  # number of rows, number of columns for 384
        yticks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
        xticks = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                  '19', '20', '21', '22', '23', '24']
    else:
        print(
            'Unknown plate format - cannot generate the heatmaps. Must be 384 well format, with no missing values')

    plate_view = plate.values.reshape(plate_reshape)

    sns.heatmap(plate_view,
                yticklabels=yticks,
                xticklabels=xticks,
                square=True,
                cmap='RdBu',
                cbar_kws={'label': 'Absolute Cell Count'})
    plt.title(PDX_sample + ' Plate ' + str(id))
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.tight_layout()
    plt.savefig(PDX_sample + '/figures/QC/heatmaps/' + PDX_sample + '_heatmap_plate_' + str(id) + '.svg')
    plt.clf()

for name, group in raw_df.groupby('Image_Metadata_Plate'):
    plate_heatmap(group.sort_values(by='Image_Metadata_Well'), name)

############################################
### Generate experiment-wide QC visuals
############################################
print('Exporting experiment-wide QC visuals for ' + PDX_sample + '.')
# Import plate layout and merge with raw count data
plate_map = pd.read_csv('plate_maps/APExBIO_FDA_Library_Information.csv')
df_with_info = raw_df
df_with_info['384_plate'] = df_with_info['Image_Metadata_Plate'].str[:-3] # e.g. Strip off the '-01' from 'AxB-FDA-A-01'
df_with_info = df_with_info.merge(plate_map, how='left', left_on=['384_plate', 'Image_Metadata_Well'], right_on=['New Plate ID', 'New Plate Location'])
Path(PDX_sample + '/data').mkdir(parents=True, exist_ok=True)
df_with_info.to_csv(PDX_sample + '/data/compiled_' + PDX_sample + '.csv')

# Function to plot boxplots
def plot_box(title, filename, x='Image_Metadata_Plate', y='Image_Count_LeukaemicNuclei', data=df_with_info):
    sns.boxplot(x=x, y=y, data=data, linewidth=0.75, fliersize=0.75, palette='Paired')
    plt.title(PDX_sample + title)
    plt.tight_layout()
    plt.xticks(rotation=45, horizontalalignment='right')
    plt.savefig((PDX_sample + '/figures/QC/' + PDX_sample + '_' + filename + '.png'), dpi=300)
    plt.clf()

# Plot boxplot all data, by plate
plot_box('Experiment-wide Raw Counts', 'experiment_wide_raw_counts')


# Plot experiment-wide row effects
df_with_info['Well Row'] = df_with_info['Image_Metadata_Well'].str[:1]
plot_box('Experiment-wide Row Effects', 'experiment_wide_row_effect', x='Well Row')

# Plot experiment-wide column effects
df_with_info['Well Col'] = df_with_info['Image_Metadata_Well'].str[1:]
plot_box('Experiment-wide Column Effects', 'experiment_wide_col_effect', x='Well Col')


# Plot counts for controls (+ve and -ve)
df_controls = pd.read_csv(PDX_sample + '/control_locs.csv')
df_with_info = df_with_info.merge(df_controls, how='left', left_on='Image_Metadata_Well', right_on='Well ID')
df_controls = df_with_info.loc[(df_with_info['control_name'] == 'Control vehicle (DMSO)') | (df_with_info['control_name'] == '+ve control - AraC') | (df_with_info['control_name'] == '+ve control - Doxorubicin')]

sns.catplot(x='control_name',
            y='Image_Count_LeukaemicNuclei',
            hue='Image_Metadata_Plate',
            data=df_with_info,
            kind='bar',
            height=3,
            aspect=4,
            capsize=.01,
            errwidth=0.6,
            palette='Paired')
plt.xlabel('Compound Type')
plt.ylabel('Raw Count')
plt.title('Raw counts by control type')
plt.tight_layout()
plt.savefig(PDX_sample + '/figures/QC/' + PDX_sample + '_raw_counts_by_control.png', dpi=300)
plt.clf()

############################################
### Generate Z' scores per plate
############################################
print('Exporting QC statistics for ' + PDX_sample + '.')
# Generate descriptive statistics for each control type
stats_df = df_controls.groupby(['Image_Metadata_Plate', 'control_name'])['Image_Count_LeukaemicNuclei'].agg(['mean', 'std', 'median', 'mad']).unstack()
stats_df.columns = [' '.join(col).strip() for col in stats_df.columns.values]  # Flatten hierarchical index
USE_CONTROL = str('+ve control - ' + USE_CONTROL)

# Calculate Z' score per plate
stats_df['Z_factor'] = (1 - (
        (3 * (stats_df['std ' + USE_CONTROL] + stats_df['std Control vehicle (DMSO)'])) / (stats_df['mean Control vehicle (DMSO)'] - stats_df['mean ' + USE_CONTROL]))).round(3)

# Calculate Z' score robust per plate
stats_df['Z_factor_robust'] = (1 - ((3 * (stats_df['mad Control vehicle (DMSO)'] + stats_df['mad ' + USE_CONTROL])) / (
        stats_df['median Control vehicle (DMSO)'] - stats_df['median ' + USE_CONTROL]))).round(3)

Path(PDX_sample + '/figures/QC/stats/').mkdir(parents=True, exist_ok=True)
stats_df.to_csv(PDX_sample + '/figures/QC/stats/' + PDX_sample + '_control_stats.csv')