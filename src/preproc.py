"""
preproc code
"""
import os
import warnings
import pandas as pd
import category_encoders as ce
from tqdm import tqdm
from .utils import load_data
warnings.filterwarnings("ignore")

class PreProc:
    """
    Preprocessing class
    """
    def __init__(self, dn_path, bg_path):
        self.dn_df = load_data(dn_path)
        self.bg_df = load_data(bg_path)
        self.date_mapping = {'109': '2020', '110': '2021', '111': '2022', '/': '-'}


    def type_transfer(self) -> pd.DataFrame:
        """
        func string
        """
        self.dn_df['捐款日期'] = self.dn_df['捐款日期'].replace(self.date_mapping, regex = True)
        self.dn_df['捐款日期'] = pd.to_datetime(self.dn_df['捐款日期'])
        self.dn_df['year'] = self.dn_df['捐款日期'].dt.year
        self.dn_df['month'] = self.dn_df['捐款日期'].dt.month
        self.dn_df['day'] = self.dn_df['捐款日期'].dt.day
        self.dn_df['is_weekday'] = (self.dn_df['捐款日期'].dt.weekday < 5).astype(int) # 是否為工作日
        return self.dn_df


    def clean_columns(self) -> pd.DataFrame:
        """
        1. 支持勸募方案： 去掉年份，例如"109-家暴"→"家暴"
        2. 捐款支付方式： 保留'-'左邊的文字，例如"公益平台-Line公益"→"公益平台"
        """
        self.dn_df['支持勸募方案_clean'] = self.dn_df['支持勸募方案'].apply(lambda x: x.split('-')[1] if '-' in x else x)
        self.dn_df['捐款支付方式_clean'] = self.dn_df['捐款支付方式'].apply(lambda x: x.split('-')[0] if '-' in x else x)
        return self.dn_df


    def donate_money(self) -> pd.DataFrame:
        """ 
        捐款金額
        """
        self.dn_df = self.dn_df[self.dn_df['捐款金額']>10].reset_index(drop=True)
        by_person_med = self.dn_df.groupby(['捐款人代碼', '捐款日期'])['捐款金額'].median() # 將一日多次捐款的捐款金額用中位數取代
        self.dn_df = self.dn_df.drop_duplicates(subset = ['捐款人代碼', '捐款日期']).sort_values(by = ['捐款人代碼', '捐款日期'])
        self.dn_df['捐款金額'] = by_person_med.values
        return self.dn_df


    def frequency_and_time_interval(self) -> pd.DataFrame:
        """
        func string
        """
        # 頻率
        self.dn_df['freq'] = self.dn_df.groupby('捐款人代碼')['捐款人代碼'].transform('count')
        self.dn_df['curr_freq'] = self.dn_df.groupby('捐款人代碼').cumcount() + 1 # 當前捐款次數
        # 時間戳記
        self.dn_df['timestamp'] = self.dn_df['捐款日期'].apply(lambda x: pd.Timestamp(x).timestamp()) / 86400
        # 上/下一次捐款日期/間隔
        self.dn_df['last_donation_date'] = \
            self.dn_df.groupby('捐款人代碼')['捐款日期'].transform(lambda x: x.sort_values(ascending=True).shift(1))
        self.dn_df['last_donation_interval'] = \
            (self.dn_df['捐款日期'] - pd.to_datetime(self.dn_df['last_donation_date'])).dt.days
        self.dn_df['next_donation_date'] = \
            self.dn_df.groupby('捐款人代碼')['捐款日期'].transform(lambda x: x.sort_values(ascending=True).shift(-1))
        self.dn_df['next_donation_interval'] = \
            (pd.to_datetime(self.dn_df['next_donation_date']) - self.dn_df['捐款日期']).dt.days
        return self.dn_df


    def month_donate(self) -> pd.DataFrame:
        """
        每月捐款：當月捐款第一次，及前一個月有捐款
        """
        month_do = self.dn_df.copy()
        month_do = month_do.sort_values(by = ['捐款人代碼', '捐款日期'])
        month_do['last_month'] = month_do.groupby('捐款人代碼')['month'].transform(lambda x: x.shift(1))
        month_do['next_month'] = month_do.groupby('捐款人代碼')['month'].transform(lambda x: x.shift(-1))
        month_do['last_month_diff'] = month_do['month'] - month_do['last_month']
        month_do['next_month_diff'] = month_do['next_month'] - month_do['month']
        month_do['month_cum'] = month_do.groupby(['捐款人代碼', 'year', 'month']).cumcount() + 1
        monthly = []
        for  i in tqdm(range(len(month_do))):
            if (
                month_do['month_cum'].iloc[i] == 1 and month_do['next_month_diff'].iloc[i] == 1
                ) or (
                month_do['month_cum'].iloc[i] == 1 and month_do['next_month_diff'].iloc[i] == 0
                ) or (
                month_do['month_cum'].iloc[i] == 1 and month_do['last_month_diff'].iloc[i] == -11
                ) or (
                month_do['month_cum'].iloc[i] == 1 and month_do['last_month_diff'].iloc[i] == 0
                ) or (
                month_do['month_cum'].iloc[i] == 1 and month_do['last_month_diff'].iloc[i] == 1
                ):
                monthly.append("1")
            else:
                monthly.append("0")
        month_do['monthly_donation'] = monthly
        month_do = month_do[['捐款人代碼','捐款收據編號','monthly_donation']]
        self.dn_df = self.dn_df.merge(month_do, on=['捐款人代碼','捐款收據編號'], how='left')
        return self.dn_df


    def donation_stats(self) -> pd.DataFrame:
        """
        之前捐款金額的平均/中位數/標準差
        """
        money_grouped = self.dn_df.groupby('捐款人代碼')['捐款金額']
        self.dn_df['prev_money_avg'] = money_grouped.expanding().mean().shift(1).reset_index(level=0, drop=True)
        self.dn_df['prev_money_med'] = money_grouped.expanding().median().shift(1).reset_index(level=0, drop=True)
        self.dn_df['prev_money_std'] = money_grouped.expanding().std(ddof=0).shift(1).reset_index(level=0, drop=True)
        return self.dn_df


    def encoding(self):
        """
        One Hot Encoding
        """
        self.dn_df = pd.get_dummies(self.dn_df, columns=['is_weekday'], prefix='is_weekday_OHE', dummy_na=False)

        # Leave-One-Out Encoding
        encoder = ce.LeaveOneOutEncoder(cols=['支持勸募方案_clean'], sigma = 0.05)
        self.dn_df['支持勸募方案_LOOE'] = \
            encoder.fit_transform(self.dn_df['支持勸募方案_clean'], self.dn_df['next_donation_interval'])
        encoder = ce.LeaveOneOutEncoder(cols=['捐款支付方式_clean'], sigma = 0.05)
        self.dn_df['捐款支付方式_LOOE'] = \
            encoder.fit_transform(self.dn_df['捐款支付方式_clean'], self.dn_df['next_donation_interval'])
        # Beta Target Encoding
        target_median = self.dn_df.groupby('支持勸募方案_clean')['next_donation_interval'].median()
        self.dn_df['支持勸募方案_BTE'] = self.dn_df['支持勸募方案_clean'].map(target_median)

        target_median = self.dn_df.groupby('捐款支付方式_clean')['next_donation_interval'].median()
        self.dn_df['捐款支付方式_BTE'] = self.dn_df['捐款支付方式_clean'].map(target_median)

        return self.dn_df


    def main(self) -> pd.DataFrame:
        """
        main function
        """
        dn_df = self.type_transfer()
        dn_df = self.clean_columns()
        dn_df = self.donate_money()
        dn_df = self.frequency_and_time_interval()
        dn_df = self.month_donate()
        dn_df = self.donation_stats()
        dn_df = self.encoding()
        return dn_df


if __name__ == "__main__":
    preproc = PreProc(
        f'{os.getcwd()}/data/現代婦女基金會捐款紀錄資料.xlsx',
        f'{os.getcwd()}/data/現代婦女基金會捐款人背景資料.xlsx'
    )
    data = preproc.main()
    print(data.head())
