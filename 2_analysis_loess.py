import pandas as pd
from pathlib import Path
from scipy import stats, linalg
from math import ceil
import numpy as np
from bokeh.models import ColumnDataSource, Span, NumeralTickFormatter
from bokeh.models.tools import HoverTool, PanTool
from bokeh.plotting import figure, output_file, show
from bokeh.transform import linear_cmap
from bokeh.io import export_png

print('Enter the name of the sample you would like to analyse:')
PDX_sample = input()

############################################
### Calculate row effects with Welch t-tests on raw plate data
############################################

raw_df = pd.read_csv(PDX_sample + '/raw_data/' + PDX_sample + '_absolute_counts_collated.csv')
rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I','J','K', 'L', 'M', 'N', 'O', 'P']
stats_df = pd.DataFrame(columns=['plate', 'row', 'statistics', 'p-value'])

for name, group in raw_df.groupby('Image_Metadata_Plate'):
    for r in rows:

        # stats for current row
        current_row_mean = group.copy().set_index('Image_Metadata_Well').filter(regex = '^((' + r + ').*)*$', axis = 0)[['Image_Count_LeukaemicNuclei']].mean()[0]
        current_row_std = group.copy().set_index('Image_Metadata_Well').filter(regex='^((' + r + ').*)*$', axis=0)[['Image_Count_LeukaemicNuclei']].std()[0]
        current_row_n = group.copy().set_index('Image_Metadata_Well').filter(regex='^((' + r + ').*)*$', axis=0).shape[0]

        # stats for all other rows
        other_row_mean = group.copy().set_index('Image_Metadata_Well').filter(regex = '^((?!' + r + ').*)*$', axis = 0)[['Image_Count_LeukaemicNuclei']].mean()[0]
        other_row_std = group.copy().set_index('Image_Metadata_Well').filter(regex='^((?!' + r + ').*)*$', axis=0)[['Image_Count_LeukaemicNuclei']].std()[0]
        other_row_n = group.copy().set_index('Image_Metadata_Well').filter(regex='^((?!' + r + ').*)*$', axis=0).shape[0]

        # Equal population variance is not assumed so Welch's t-test is performed.
        t_score = stats.ttest_ind_from_stats(mean1=current_row_mean, std1=current_row_std, nobs1=current_row_n, mean2=other_row_mean, std2=other_row_std, nobs2=other_row_n, equal_var=False)
        stats_df = stats_df.append({'plate': name, 'row': r, 'statistics': round(t_score[0],2), 'p-value': round(t_score[1], 4)}, ignore_index=True)

# State if systematic row effects have been detected
if stats_df['p-value'].min() > 0.05:
    print(f'No systematic errors have been detected in the {PDX_sample} dataset and normalisation may not be appropriate.')
else:
    print(f'Systematic errors have been detected in the {PDX_sample} dataset and normalisation should be applied.')

# Export the results from the row-by-row Welch's t-tests for each plate
Path(PDX_sample + '/data/stats').mkdir(parents=True, exist_ok=True)
stats_df.to_csv(PDX_sample + '/data/stats/' + PDX_sample + '_row_effects.csv')


############################################
### Calculate and apply LOESS normalisation
############################################

def lowess(x, y, f=0.05, iter=10):
    """lowess(x, y, f=2./3., iter=3) -> yest
    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.
    """
    n = len(x)
    r = int(ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest

def normalize_b(df, feature):
    """Takes DataFrame with measurements and feature name and adds a column with normalized values of the feature.
    """
    mean = df[[feature]].median()
    std = df[[feature]].std()
    df[feature + '_norm'] = df[feature].apply(lambda x:(x - mean)/std)
    return(df)


positive_control_pos = ['J23', 'J24', 'K23', 'K24'] # Should not be hardcoded - positive controls provided already

df_loess = pd.DataFrame(columns=['Image_Count_LeukaemicNuclei', 'Image_Count_MSCNuclei', 'Image_Metadata_PDX', 'Image_Metadata_Plate', 'Image_Metadata_Well'])

for name, group in raw_df.groupby('Image_Metadata_Plate'):
    PDX_array = group['Image_Count_LeukaemicNuclei'].to_numpy()
    compound_no = group.index.to_numpy()
    group['loess_estimation'] = lowess(compound_no, PDX_array, f=0.05)
    group['loess_residual'] = group.Image_Count_LeukaemicNuclei - group.loess_estimation
    normalize_b(group, 'loess_residual')

    median_positive_control = group[group['Image_Metadata_Well'].isin(positive_control_pos)]['loess_residual_norm'].median()
    group['percent_of_pos_control'] = (group['loess_residual_norm'] / median_positive_control) * 100
    group['viability'] = 100 - group['percent_of_pos_control']

    df_loess = df_loess.append(group)

# Enclose in try-except block to catch any NA errors when running. Alert and give option to fill with dataset median.


############################################
### Merge with drug info and calculate mean±SD for each
############################################

# If Image_Metadata_Well in single digit format, correct (e.g. A1 to A01)
df_loess['Row'] = df_loess['Image_Metadata_Well'].str.extract(r'(^\w)')
df_loess['Col'] = df_loess['Image_Metadata_Well'].str.extract(r'^\w(\d{1,2})')
df_loess['Col'] = df_loess['Col'].str.zfill(2)
df_loess['Image_Metadata_Well'] = df_loess['Row'].map(str) + df_loess['Col'].map(str)

# Import plate layout and merge with raw count data
plate_map = pd.read_csv('plate_maps/APExBIO_FDA_Library_Information.csv').fillna(1) # Fillna otherwise pivot table removes any rows with NAs.
df_with_info = df_loess
df_with_info['384_plate'] = df_with_info['Image_Metadata_Plate'].str[:-3] # e.g. Strip off the '-01' from 'AxB-FDA-A-01'
df_with_info = df_with_info.merge(plate_map, how='left', left_on=['384_plate', 'Image_Metadata_Well'], right_on=['New Plate ID', 'New Plate Location'])
df_with_info.dropna(subset=['Item Name'], inplace=True) # Drop empty wells
df_with_info.to_csv(PDX_sample + '/data/compiled_normalised_' + PDX_sample + '.csv')

# Pivot to have aggregated (mean±SD) for each unique drug
df_aggregated = pd.pivot_table(df_with_info, values=['viability'], index=['Item Name', 'CatalogNumber', 'CAS Number', 'Pathway', 'Target', 'Information', 'SMILES'], aggfunc=[np.mean, np.std])
df_aggregated.columns = [' '.join(col).strip() for col in df_aggregated.columns.values]  # Flatten hierarchical index
df_aggregated['mean viability'], df_aggregated['std viability'] = df_aggregated['mean viability'].round(1), df_aggregated['std viability'].round(1)
df_aggregated.to_csv(PDX_sample + '/data/' + PDX_sample + '_normalised_aggregated.csv')

############################################
### Calculate and export hits for dataset
############################################

# Calculate 2SD below the mean of the dataset (excl. controls) to define 'hits'
dataset_mean = df_aggregated['mean viability'].mean()
dataset_std = df_aggregated['mean viability'].std()
dataset_2std = dataset_mean - 2 * dataset_std
print(f'Anything below {round(dataset_2std, 2)}% relative viability is considered a hit in this dataset.')

# Define and export 'hits' to CSV
dataset_hits = df_aggregated[df_aggregated['mean viability'] <= dataset_2std]
print(f'There are {dataset_hits["mean viability"].count()} hits, with a hit rate of {((dataset_hits["mean viability"].count() / df_aggregated["mean viability"].count())*100).round(1)}%.')
dataset_hits.to_csv(PDX_sample + '/data/' + PDX_sample + '_normalised_hits.csv')

############################################
### Plot interactive scatter of all points in the dataset
############################################

df_aggregated = df_aggregated.reset_index()

output_file(PDX_sample + '/figures/' + PDX_sample + '_all_normalised_scatterplot.html', title=(PDX_sample + ' Screening Overview'))
source = ColumnDataSource(data=df_aggregated)
p = figure(title=('Screening overview for ' + PDX_sample), x_axis_label='Library Compounds', y_axis_label='Relative response (%)', plot_width=1200, plot_height=600, y_axis_type='log')
mapper = linear_cmap(field_name='mean viability', palette=['red', 'grey'], low=dataset_2std, high=dataset_2std)
p.scatter(x='index', y='mean viability', source=source, size=7, alpha=0.5, color=mapper)
hline = Span(location=dataset_2std, dimension='width', line_color='gray', line_width=2, line_alpha=0.5, line_dash='dashed')
p.add_layout(hline)
p.xaxis.ticker = []
p.toolbar.logo = None
p.toolbar.autohide = True
p.add_tools(PanTool(dimensions="height"))
p.yaxis[0].formatter = NumeralTickFormatter(format='0')
p.ygrid.grid_line_alpha = 0.5
hover = HoverTool()
hover.tooltips=[
    ('Compound Name', '@{Item Name}'),
    ('Pathway', '@Pathway'),
    ('Target', '@Target'),
    ('CAS Number', '@{CAS Number}'),
    ('Relative response', '@{mean viability}'),
    ('Response SD', '@{std viability}'),
    ('Information', '@Information')

]
p.add_tools(hover)


# Sortable and linked datatable of results
from bokeh.models import DataTable, TableColumn
from bokeh.layouts import column
datatable_cols = [
    TableColumn(field='Item Name', title='Compound name'),
    TableColumn(field='mean viability', title='Relative response (%)'),
    TableColumn(field='Pathway', title='Pathway'),
    TableColumn(field='Target', title='Target'),
    TableColumn(field='Information', title='Information')
    ]
data_table = DataTable(source=source, columns=datatable_cols, frozen_columns=1, width=1200, height_policy='max', width_policy='max')

#export_png(p, filename=('figures/' + PDX_sample + '_all_normalised_scatterplot.png'))
show(column(p, data_table))