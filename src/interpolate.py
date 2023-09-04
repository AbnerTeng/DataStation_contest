"""
interpolate code
"""
import os
import warnings
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from .preproc import PreProc
warnings.filterwarnings("ignore")

class LocInterpolate(PreProc):
    """
    Location interpolate
    """
    def __init__(self):
        """
        init function
        """
        super().__init__(
            f'{os.getcwd()}/data/現代婦女基金會捐款紀錄資料.xlsx',
            f'{os.getcwd()}/data/現代婦女基金會捐款人背景資料.xlsx'
        )
        self.model = {
            'rf': RandomForestClassifier(),
            'svm': SVC()
        }
        self.agg_dict = {
            'freq': 'mean',
            '支持勸募方案_clean': lambda x: x.mode().iloc[0],
            '捐款支付方式_clean': lambda x: x.mode().iloc[0],
            '捐款金額': ['quantile', 'mean', lambda x: x.std(ddof=0)],
            'timestamp': ['quantile', 'mean', lambda x: x.std(ddof=0)],
            'monthly_donation': 'mean',
            '支持勸募方案_LOOE': 'mean',
            '捐款支付方式_LOOE': 'mean',
            '支持勸募方案_BTE': 'mean',
            '捐款支付方式_BTE': 'mean',
            'is_weekday_OHE_0': 'mean',
            'is_weekday_OHE_1': 'mean'
        }
        self.column_mapping = {
            'freq': 'freq_mean',
            '支持勸募方案_clean': '支持勸募方案_clean_mode',
            '捐款支付方式_clean': '捐款支付方式_clean_mode',
            '捐款金額_quantile': '捐款金額_quantile',
            '捐款金額_mean': '捐款金額_mean',
            '捐款金額_<lambda>': '捐款金額_std',
            'timestamp_quantile': 'timestamp_quantile',
            'timestamp_mean': 'timestamp_mean',
            'timestamp_<lambda>': 'timestamp_std',
            'monthly_donation_mean': 'monthly_donation_mean',
            '支持勸募方案_LOOE_mean': '支持勸募方案_LOOE_mean',
            '捐款支付方式_LOOE_mean': '捐款支付方式_LOOE_mean',
            '支持勸募方案_BTE_mean': '支持勸募方案_BTE_mean',
            '捐款支付方式_BTE_mean': '捐款支付方式_BTE_mean',
            'is_weekday_OHE_0_mean': 'is_weekday_OHE_0_mean',
            'is_weekday_OHE_1_mean': 'is_weekday_OHE_1_mean'
        }
        self.dn_df = super().main()


    def group(self):
        """
        function string
        """
        grouped_dn_df = self.dn_df.groupby('捐款人代碼').agg(self.agg_dict)
        grouped_dn_df.reset_index(inplace = True)
        grouped_dn_df.columns = ['捐款人代碼'] + [col[0] + '_' + col[1] \
            if isinstance(col, tuple) else col for col in grouped_dn_df.columns[1:]]
        grouped_dn_df.rename(columns = self.column_mapping, inplace = True)

        ## one hot encoding
        dummy_columns = ['支持勸募方案_clean_<lambda>', '捐款支付方式_clean_<lambda>']
        grouped_dn_df = pd.get_dummies(
            grouped_dn_df, columns = dummy_columns,
            prefix = ['ohe', 'ohe'], dummy_na = False
        )
        return grouped_dn_df


    def join(
            self,
            donate_data: pd.DataFrame,
            target_col1: str,
            target_col2: str
        ):
        """
        Join dataframes
        """
        background = self.bg_df
        join_bg_df = background.merge(donate_data, how = 'inner', on = '捐款人代碼')
        join_bg_df[target_col1] = join_bg_df[target_col2].apply(lambda x: str(x)[0])
        join_bg_df = join_bg_df.drop(columns=["性別", "居住縣市郵遞區號", "生日年月"])
        join_bg_df = join_bg_df.dropna()
        join_bg_df = join_bg_df[join_bg_df['freq_mean'] != 1].reset_index(drop = True)
        notna_df = join_bg_df[join_bg_df[target_col1] != 'n']
        isna_df = join_bg_df[join_bg_df[target_col1] == 'n']
        return notna_df, isna_df


    def train_model(
            self,
            notna: pd.DataFrame,
            isna: pd.DataFrame,
            target_col: str,
            data: pd.DataFrame
        ):
        """
        Use classifier to interpolate
        """
        x_data = notna.drop([target_col, '捐款人代碼'], axis = 1)
        if target_col == 'city':
            label_encoder = LabelEncoder()
            y_data = label_encoder.fit_transform(notna[target_col])
            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3, random_state = 42)
            model = self.model['rf']
        else:
            label_encoder = LabelEncoder()
            y_data = label_encoder.fit_transform(notna[target_col])
            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3)
            model = self.model['svm']
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("=========================================")
        print(f"Model Accuracy = {accuracy:.2f}")
        print("=========================================")
        y_pred = model.predict(isna.drop([target_col, '捐款人代碼'], axis = 1))
        print(y_pred)
        decoded_y_pred = label_encoder.inverse_transform(y_pred)
        isna = pd.concat([isna[['捐款人代碼']].reset_index(),
                          pd.Series(decoded_y_pred, name = target_col)], axis = 1)
        all_data = pd.concat([notna[['捐款人代碼', target_col]], isna], axis = 0).drop('index', axis = 1)
        data = data.merge(all_data, how = 'left', on = '捐款人代碼')
        return data


    def city_encoding(self, data):
        """
        encoding session
        """
        data = pd.get_dummies(data, columns=['gender'], prefix='gender_OHE', dummy_na=False)
        category_counts = data['city'].value_counts(normalize=True)
        data['city_FE'] = data['city'].map(category_counts)
        target_median = data.groupby('city')['next_donation_interval'].median()
        data['city_BTE'] = data['city'].map(target_median)
        return data


    def main(self):
        """
        main function
        """
        grouped_dn_df = self.group()
        notna_city_df, isna_city_df = self.join(
            grouped_dn_df, 'city', '居住縣市郵遞區號'
        )
        notna_gender_df, isna_gender_df = self.join(
            grouped_dn_df, 'gender', '性別'
        )
        background_city = self.train_model(
            notna_city_df, isna_city_df, 'city',
            self.bg_df
        )
        background_gender = self.train_model(
            notna_gender_df, isna_gender_df, 'gender',
            background_city
        )
        full_data = self.dn_df.merge(
            background_gender[['捐款人代碼', 'city', 'gender']],
            how = 'left', on = '捐款人代碼'
        )
        full_data = self.city_encoding(full_data)

        return full_data


if __name__ == "__main__":
    loc_interpolate = LocInterpolate()
    dat = loc_interpolate.main()
    print(dat.head())
