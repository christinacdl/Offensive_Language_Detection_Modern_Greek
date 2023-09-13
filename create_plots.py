import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 

# CLASS DISTRIBUTION OF OFFENSIVE AND NOT VALUES FOR TRAIN, DEVELOPMENT AND TEST SETS BASED ON THRESHOLD
# List of thresholds
thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

# Initialize lists to store counts
train_counts_label_1 = []
val_counts_label_1 = []
test_counts_label_1 = []
train_counts_label_0 = []
val_counts_label_0 = []
test_counts_label_0 = []

# Read each CSV file and extract counts
for threshold in thresholds:
    filename = f'train_test_files/prepared_files/threshold_run_1/{threshold}_value_counts.csv'
    df = pd.read_csv(filename, sep=',')
    train_counts_label_1.append(df.loc[1,'train_counts'])
    val_counts_label_1.append(df.loc[1,'val_counts'])
    test_counts_label_1.append(df.loc[1,'test_counts'])
    train_counts_label_0.append(df.loc[0,'train_counts'])
    val_counts_label_0.append(df.loc[0,'val_counts'])
    test_counts_label_0.append(df.loc[0,'test_counts'])

# Create a DataFrame for bar plot
df_bar_plot = pd.DataFrame({
    'Offensive Train Counts': train_counts_label_1,
    'Offensive Validation Counts': val_counts_label_1,
    'Offensive Test Counts Label ': test_counts_label_1,
    'Not Train Counts': train_counts_label_0,
    'Not Validation Counts': val_counts_label_0,
    'Not Test Counts': test_counts_label_0}, index=thresholds)

# Create bar plot
df_bar_plot.plot(kind='bar', figsize=(12,6), color=['#A2142F', '#EDB120', '#D95319', '#0072BD', '#77AC30', '#7E2F8E'])
plt.title('Class Distribution based on Thresholds')
plt.ylabel('Counts')
plt.xlabel('Threshold')
plt.savefig('/metrics/Class_distribution', bbox_inches='tight')
plt.show()
plt.close()


# MACRO-F1 SCORE MODEL COMPARISON PLOT
# Open and read the metrics report 
metrics_1 = pd.read_csv('/predictions/threshold_run_4/All_Metrics_BERTMULTI.tsv', sep ='\t')
metrics_2 = pd.read_csv('/predictions/threshold_run_6/All_Metrics_GREEK_MEDIA_BERT.tsv', sep ='\t')
metrics_3 = pd.read_csv('/predictions/threshold_run_7/All_Metrics_mDEBERTa.tsv', sep ='\t')
metrics_4 = pd.read_csv('/predictions/threshold_run_9/All_Metrics_GREEK_BERT.csv', sep =',')
metrics_5 = pd.read_csv('/predictions/threshold_run_10/All_Metrics_XLM_ROBERTA.csv', sep =',')
metrics_6 = pd.read_csv('/predictions/threshold_run_5/All_Metrics_ETHICAL_EYE.tsv', sep ='\t')


# Plot the metrics to compare 
plt.plot(metrics_1['Threshold'], metrics_1['Macro_F1'], label='BERT-Multilingual-Base-Uncased',
         color='#A2142F', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='#A2142F', markersize=8)
plt.plot(metrics_2['Threshold'], metrics_2['Macro_F1'], label='Greek-Media-BERT-Base-Uncased',
         color='#0072BD', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='#0072BD', markersize=8)
plt.plot(metrics_3['Threshold'], metrics_3['Macro_F1'], label='DeBERTa-Multilingual-V3-Base',
         color='#D95319', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='#D95319', markersize=8)
plt.plot(metrics_4['Threshold'], metrics_4['Macro_F1'], label='Greek-BERT-Base-Uncased-V1',
         color='#77AC30', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='#77AC30', markersize=8)
plt.plot(metrics_5['Threshold'], metrics_5['Macro_F1'], label='XLM-RoBERTa-Base',
         color='#7E2F8E', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='#7E2F8E', markersize=8)
plt.xlabel('Threshold')
plt.legend()
plt.grid()
plt.ylabel('Macro-F1 Score')
plt.title('Model Macro-F1 Scores Based on Thresholds')
plt.savefig('/metrics/Model_Comparison_Thresholds', bbox_inches='tight')
plt.show()
plt.close()
