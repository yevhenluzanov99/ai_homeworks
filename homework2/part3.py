import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Upload homework2_data.csv
df = pd.read_csv('homework2\homework2_data.csv')

#You can extract each column as an array:
#Take a look at data in each column.
delta_frame_time = df['delta_frame_time'].values
dns_answers_ttl = df['dns_answers_ttl'].values
is_cc_provider = df['is_cc_provider'].values
is_ip_private = df['is_ip_private'].values
delay = df['delay'].values


#What type of data each column contains (ordinal, nominal, discerete or continuous)?
'''
Create 5 hist in one plot with info about each column-feature
'''
fig, axs = plt.subplots(3, 2, figsize=(10, 8))

#feature delta_frame_time
axs[0, 0].hist(delta_frame_time, bins=30, alpha=0.5, color='r')
axs[0, 0].set_title('delta_frame_time has type continuous')
median_value_delta = np.median(delta_frame_time)
mean_value_delta = np.mean(delta_frame_time)
axs[0, 0].axvline(median_value_delta, color='blue', linestyle='dashed', linewidth=1)
axs[0, 0].text(median_value_delta, 0, f'Median: {median_value_delta}', color='blue', rotation=90, ha = 'right')
axs[0, 0].axvline(mean_value_delta, color='red', linestyle='dashed', linewidth=1)
axs[0, 0].text(mean_value_delta, 0, f'Mean: {mean_value_delta}', color='red', rotation=90, ha = 'left')

#feature dns_answers_ttl
axs[0, 1].hist(dns_answers_ttl, bins=30, alpha=0.5, color='g')
axs[0, 1].set_title('dns_answers_ttl has type discrete')
median_value_dns_answers = np.median(dns_answers_ttl)
mean_value_dns_answers = np.mean(dns_answers_ttl)
axs[0, 1].axvline(median_value_dns_answers, color='blue', linestyle='dashed', linewidth=1)
axs[0, 1].text(median_value_dns_answers, 0, f'Median: {median_value_dns_answers}', color='blue', rotation=90, ha = 'right')
axs[0, 1].axvline(mean_value_dns_answers, color='red', linestyle='dashed', linewidth=1)
axs[0, 1].text(mean_value_dns_answers, 0, f'Mean: {mean_value_dns_answers}', color='red', rotation=90, ha = 'left')

#feature is_cc_provider
axs[1, 0].hist(is_cc_provider, bins=30, alpha=0.5, color='b')
axs[1, 0].set_title('is_cc_provider has type nominal')
median_value_cc_provider = np.median(is_cc_provider)
mean_value_cc_provider = np.mean(is_cc_provider)
axs[1, 0].axvline(median_value_cc_provider, color='blue', linestyle='dashed', linewidth=1)
axs[1, 0].text(median_value_cc_provider, 0, f'Median: {median_value_cc_provider}', color='blue', rotation=90, ha = 'right')
axs[1, 0].axvline(mean_value_cc_provider, color='red', linestyle='dashed', linewidth=1)
axs[1, 0].text(mean_value_cc_provider, 0, f'Mean: {mean_value_cc_provider}', color='red', rotation=90, ha = 'left')

#feature is_ip_private
axs[1, 1].hist(is_ip_private, bins=30, alpha=0.5, color='y')
axs[1, 1].set_title('is_ip_private has type nominal')
median_value_ip_private = np.median(is_ip_private)
mean_value_ip_private = np.mean(is_ip_private)
axs[1, 1].axvline(median_value_ip_private, color='blue', linestyle='dashed', linewidth=1)
axs[1, 1].text(median_value_ip_private, 0, f'Median: {median_value_ip_private}', color='blue', rotation=90, ha = 'right')
axs[1, 1].axvline(mean_value_ip_private, color='red', linestyle='dashed', linewidth=1)
axs[1, 1].text(mean_value_ip_private, 0, f'Mean: {mean_value_ip_private}', color='red', rotation=90, ha = 'left')

#feature delay
axs[2, 0].hist(delay, bins=30, alpha=0.5, color='m')
axs[2, 0].set_title('delay has type continuous ')
median_value_delay = np.median(delay)
mean_value_delay = np.mean(delay)
axs[2, 0].axvline(median_value_delay, color='blue', linestyle='dashed', linewidth=1)
axs[2, 0].text(median_value_delay, 0, f'Median: {median_value_delay}', color='blue', rotation=90, ha = 'right')
axs[2, 0].axvline(mean_value_delay, color='red', linestyle='dashed', linewidth=1)
axs[2, 0].text(mean_value_delay, 0, f'Mean: {mean_value_delay}', color='red', rotation=90, ha = 'left')


plt.tight_layout() 
plt.show()