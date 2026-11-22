import io
import unittest

import numpy as np
import pandas as pd

from forecast_data import load_table, prepare_series


class NamedBytesIO(io.BytesIO):
    def __init__(self, data=b"", name="data.csv"):
        super().__init__(data)
        self.name = name


class ForecastDirtyInputTests(unittest.TestCase):
    def csv(self, text, name="data.csv"):
        return NamedBytesIO(text.encode("utf-8"), name=name)

    def test_bom_semicolon_blank_rows_and_whitespace_headers(self):
        src=self.csv(
            "\ufeff date ; return ; note\n"
            "2024-01-01 ; 0.01 ; a\n"
            "\n"
            "2024-02-01 ; 0.02 ; b\n"
            "2024-03-01 ; -0.01 ; c\n"
            "2024-04-01 ; 0.015 ; d\n"
        )
        df=load_table(src)
        self.assertEqual(list(df.columns),["return","note"])
        s,w=prepare_series(df,"return","return")
        self.assertEqual(len(s),4)
        self.assertEqual(w,[])

    def test_transposed_input_is_supported(self):
        src=self.csv(
            "date,2024-01-01,2024-02-01,2024-03-01,2024-04-01\n"
            "return,0.01,0.02,-0.01,0.015\n"
        )
        df=load_table(src)
        self.assertEqual(df.attrs["orientation"],"transposed")
        s,w=prepare_series(df,"return","return")
        self.assertEqual(len(s),4)
        self.assertTrue(any("転置" in x for x in w))

    def test_unsorted_dates_are_sorted_with_warning(self):
        src=self.csv(
            "date,return\n"
            "2024-03-01,0.03\n"
            "2024-01-01,0.01\n"
            "2024-02-01,0.02\n"
            "2024-04-01,0.04\n"
        )
        df=load_table(src)
        self.assertTrue(df.index.is_monotonic_increasing)
        s,w=prepare_series(df,"return","return")
        self.assertEqual(list(s.values),[0.01,0.02,0.03,0.04])
        self.assertTrue(any("並べ替え" in x for x in w))

    def test_duplicate_dates_are_rejected(self):
        src=self.csv(
            "date,return\n"
            "2024-01-01,0.01\n"
            "2024-01-01,0.02\n"
            "2024-02-01,0.03\n"
        )
        with self.assertRaisesRegex(ValueError,"重複日付"):
            load_table(src)

    def test_internal_missing_or_text_value_is_rejected_not_compressed(self):
        for bad in ["", "abc", "inf"]:
            src=self.csv(
                "date,return\n"
                "2024-01-01,0.01\n"
                f"2024-02-01,{bad}\n"
                "2024-03-01,0.03\n"
                "2024-04-01,0.04\n"
            )
            df=load_table(src)
            with self.assertRaisesRegex(ValueError,"途中に欠損または不正値"):
                prepare_series(df,"return","return")

    def test_leading_and_trailing_missing_values_are_trimmed(self):
        src=self.csv(
            "date,return\n"
            "2023-12-01,\n"
            "2024-01-01,0.01\n"
            "2024-02-01,0.02\n"
            "2024-03-01,0.03\n"
            "2024-04-01,\n"
        )
        df=load_table(src)
        s,_=prepare_series(df,"return","return")
        self.assertEqual(len(s),3)

    def test_invalid_prices_and_impossible_returns_are_rejected(self):
        prices=load_table(self.csv(
            "date,price\n"
            "2024-01-01,100\n"
            "2024-02-01,0\n"
            "2024-03-01,105\n"
            "2024-04-01,110\n"
        ))
        with self.assertRaisesRegex(ValueError,"0以下"):
            prepare_series(prices,"price","price")

        returns=load_table(self.csv(
            "date,return\n"
            "2024-01-01,0.01\n"
            "2024-02-01,-1.2\n"
            "2024-03-01,0.02\n"
            "2024-04-01,0.03\n"
        ))
        with self.assertRaisesRegex(ValueError,"-100%未満"):
            prepare_series(returns,"return","return")

    def test_extreme_return_is_warned_but_not_silently_deleted(self):
        dates=pd.date_range("2020-01-01",periods=80,freq="D")
        vals=np.array([0.01+((i%5)-2)*0.001 for i in range(80)],dtype=float)
        vals[40]=0.50
        text="date,return\n"+"\n".join(f"{d.date()},{v}" for d,v in zip(dates,vals))+"\n"
        df=load_table(self.csv(text))
        s,w=prepare_series(df,"return","return")
        self.assertEqual(len(s),80)
        self.assertTrue(any("5標準偏差" in x for x in w))

    def test_excel_input_uses_same_validation(self):
        bio=NamedBytesIO(name="sample.xlsx")
        raw=pd.DataFrame({
            "date":pd.date_range("2024-01-01",periods=4,freq="MS"),
            "return":[0.01,0.02,-0.01,0.015],
        })
        with pd.ExcelWriter(bio,engine="openpyxl") as writer:
            raw.to_excel(writer,index=False)
        bio.seek(0)
        df=load_table(bio)
        s,_=prepare_series(df,"return","return")
        self.assertEqual(len(s),4)


if __name__=="__main__":
    unittest.main()
