import pandas as pd
import numpy as np
from scipy import stats

class ABHypothesis:
    def __init__(self, data):
        """
        Initialize the class with the dataset.
        """
        self.data = data

    def segment_data(self, feature, value=None, exclude_values=None):
        """
        Segment the data based on a feature. Optionally filter by value or exclude certain values.
        """
        if exclude_values is not None:
            data_segment = self.data[~self.data[feature].isin(exclude_values)]
        else:
            data_segment = self.data.copy()

        if value is not None:
            data_segment = data_segment[data_segment[feature] == value]
        
        return data_segment

    def check_identical_values(self, metric):
        """
        Check if all values for a metric are identical.
        """
        unique_values = self.data[metric].dropna().unique()
        return len(unique_values) == 1

    def chi_squared_test(self, feature, metric):
        """
        Perform chi-squared test for categorical data.
        """
        contingency_table = pd.crosstab(self.data[feature], self.data[metric])
        chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
        return chi2, p_value

    def t_test(self, group_a, group_b, metric):
        """
        Perform a t-test between two groups on a given metric.
        """
        if self.check_identical_values(metric):
            print(f"Warning: All values for {metric} are identical. Skipping t-test.")
            return None, None

        t_stat, p_value = stats.ttest_ind(group_a[metric].dropna(), group_b[metric].dropna(), nan_policy='omit')
        return t_stat, p_value

    def z_test(self, group_a, group_b, metric):
        """
        Perform a z-test between two groups if sample size is large (>30).
        """
        mean_a, mean_b = group_a[metric].mean(), group_b[metric].mean()
        std_a, std_b = group_a[metric].std(), group_b[metric].std()
        n_a, n_b = group_a[metric].count(), group_b[metric].count()

        z_stat = (mean_a - mean_b) / np.sqrt((std_a**2 / n_a) + (std_b**2 / n_b))
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        return z_stat, p_value

    def interpret_p_value(self, p_value, alpha=0.05):
        """
        Interpret the null hypothesis based on the p-value.
        """
        if p_value is None:
            return "Test skipped due to identical values."
        return "Reject the null hypothesis." if p_value < alpha else "Fail to reject the null hypothesis."

    def risk_across_provinces(self):
        """
        Test for risk differences across provinces using Chi-Squared test on TotalPremium.
        """
        chi2, p_value = self.chi_squared_test('Province', 'TotalPremium')
        return f"Chi-squared test on Province and TotalPremium: chi2 = {chi2}, p-value = {p_value}\n" + self.interpret_p_value(p_value)

    def risk_between_postalcodes(self):
        """
        Test for risk differences between postal codes using Chi-Squared test.
        """
        chi2, p_value = self.chi_squared_test('PostalCode', 'TotalPremium')
        return f"Chi-squared test on PostalCode and TotalPremium: chi2 = {chi2}, p-value = {p_value}\n" + self.interpret_p_value(p_value)

    def margin_between_postalcodes(self):
        """
        Test for margin differences between postal codes using t-test or z-test on TotalPremium.
        """
        postal_codes = self.data['PostalCode'].unique()
        if len(postal_codes) < 2:
            return "Not enough unique postal codes for testing."

        group_a = self.segment_data('PostalCode', value=postal_codes[0])
        group_b = self.segment_data('PostalCode', value=postal_codes[1])
        
        if len(group_a) > 30 and len(group_b) > 30:
            z_stat, p_value = self.z_test(group_a, group_b, 'TotalPremium')
            return f"Z-test on TotalPremium: Z-statistic = {z_stat}, p-value = {p_value}\n" + self.interpret_p_value(p_value)
        else:
            t_stat, p_value = self.t_test(group_a, group_b, 'TotalPremium')
            return f"T-test on TotalPremium: T-statistic = {t_stat}, p-value = {p_value}\n" + self.interpret_p_value(p_value)

    def risk_between_genders(self):
        """
        Test for risk differences between Men and Women using t-test on TotalPremium.
        """
        self.data = self.segment_data('Gender', exclude_values=['Not Specified'])

        group_a = self.segment_data('Gender', value='Male')
        group_b = self.segment_data('Gender', value='Female')

        if group_a.empty or group_b.empty:
            return "One of the gender groups is empty. Test cannot be performed."

        t_stat, p_value = self.t_test(group_a, group_b, 'TotalPremium')
        return f"T-test on TotalPremium: T-statistic = {t_stat}, p-value = {p_value}\n" + self.interpret_p_value(p_value)

    def run_all_tests(self):
        """
        Run all hypothesis tests and return the results.
        """
        results = {
            'Risk Differences Across Provinces': self.risk_across_provinces(),
            'Risk Differences Between Postal Codes': self.risk_between_postalcodes(),
            'Margin Differences Between Postal Codes': self.margin_between_postalcodes(),
            'Risk Differences Between Women and Men': self.risk_between_genders(),
        }
        return results