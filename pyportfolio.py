# %% imports

import pandas as pd
from typing import Tuple, Any
import re
from glob import glob
from pdfminer.high_level import extract_text
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from functools import reduce

# %% functions and classes

def parseGermanDateTimes(germanDateTimeStrings: pd.Series) -> pd.Series:
    return pd.to_datetime(germanDateTimeStrings, format='%d.%m.%Y')

def parseGermanFloat(germanFloatString: str) -> float:
    return float(germanFloatString.replace(".", "").replace(",", "."))

class DkbParsingError(Exception):
    def __init__(self, pdf: "DkbPdfParser", field: str, regexp: str):
        msg = "Couldn't parse {} using RegExp {} on PDF path {}: \n\n{}"
        msg = msg.format(field, regexp, pdf.pdfFilePath, pdf.pdfText)
        super().__init__(msg)

class DkbPdfParser:
        
    fields = [
        ("Schlusstag", r'[\s\S]*Schlusstag(?:\n|\/-Zeit )([0-9]{2}\.[0-9]{2}\.[0-9]{4}).*\n[\s\S]*', parseGermanDateTimes),
        ("Ausführungskurs", r'[\s\S]*Ausführungskurs (.*) [\s\S].', parseGermanFloat),
        ("Stück", r'[\s\S]*Stück (.*)\n[\s\S].', parseGermanFloat),
        ("ISIN", r'[\s\S]*\n([a-zA-Z]{2}\w{9}[0-9])\n[\s\S]*', lambda x: x),
        ("Wertpapierbezeichnung", r'[\s\S]*Wertpapierbezeichnung\n\n(.*)\n[\s\S]*', lambda x: x),
        ("Verkauf/Kauf", r'[\s\S]* (Ausgabe|Verkauf|Kauf|Rücknahme)\s[\s\S]*', lambda x: x)
    ]
    
    def __init__(self, pdfFilePath: str) -> None:
        self.pdfFilePath = pdfFilePath
        self.pdfText = extract_text(pdfFilePath)
        
    def parse(self, field, regexp, parsingMethod) -> Any:
        try:
            match = re.match(regexp, self.pdfText)
            return parsingMethod(str(match.group(1)))
        except:
            raise DkbParsingError(self, field, regexp)
    
    def getRow(self) -> str:
        return [self.parse(*x) for x in self.fields]
    
    def getFields() -> str:
        return [field for field, _, _ in DkbPdfParser.fields]


# %% PDF parsing and preparation of numbers

pdfFilePathsWildcard = "pdf/*.pdf"

columns = DkbPdfParser.getFields()
data = [DkbPdfParser(p).getRow() for p in glob(pdfFilePathsWildcard)]
df = pd.DataFrame(columns=columns, data=data)
df.loc[df["Verkauf/Kauf"] == "Ausgabe", "Verkauf/Kauf"] = "Kauf"
df.loc[df["Verkauf/Kauf"] == "Rücknahme", "Verkauf/Kauf"] = "Verkauf"
df.loc[df["Verkauf/Kauf"] == "Verkauf", "Stück"] *= -1
copy = df.copy(deep=True)
copy.Schlusstag -= pd.to_timedelta('1d')
copy.loc[:,"Stück"] = 0.0
df = df.append(copy, ignore_index=True)
df = df.sort_values("Schlusstag")
df["Investitionssumme"] = df["Stück"] * df["Ausführungskurs"]


start = df.iloc[0].Schlusstag
end = df.iloc[-1].Schlusstag
timePeriod = end - start + pd.to_timedelta("1d")

days = [
    start + pd.to_timedelta("{}d".format(d)) 
    for d in range(timePeriod.days)
]

wertpapiere = df.Wertpapierbezeichnung.unique()

lines = pd.DataFrame(
    index=days, 
    columns=wertpapiere, 
    data=np.empty((len(days), len(wertpapiere))).fill(np.nan)
)

df["Stücksumme"] = pd.Series([0.0] * len(df))
df["Wert"] = pd.Series([0.0] * len(df))
def cumsum(values: pd.Series) -> pd.Series:
    sum = 0.0
    result = [0.0] * len(values)
    for i, x in enumerate(values):
        sum = x + sum if x + sum > 0 else 0.0
        result[i] = sum
    return result

df["Investitionssumme"] = cumsum(df["Investitionssumme"])
lines["Investitionssumme"] = np.nan
lines.loc[df.Schlusstag, "Investitionssumme"] = df.Investitionssumme.array
for bez in df.Wertpapierbezeichnung.unique():
    entries = (df.Wertpapierbezeichnung == bez)
    df.loc[entries, "Stücksumme"] = df[entries]["Stück"].cumsum()
    df.loc[entries, "Wert"] = df.loc[entries, "Stücksumme"] * df.loc[entries, "Ausführungskurs"]
    lines.loc[df[entries].Schlusstag, bez] = df[entries].Wert.array

df = df.round(2)
df.to_csv("./stocks-accounting-entries.csv", index=False)


# %% plotting

lines = lines.astype(float)
lines[lines < 0] = 0
lines = lines.interpolate(method='linear', limit_direction='both', axis=0)
lines["Investitionswert"] = lines.sum(axis=1) - lines.Investitionssumme
#lines = df.round(2)

with PdfPages('portfolio.pdf') as pdf:
    for monthsDuration in [3, 6, 12, 24]:
        days = int(round(monthsDuration * 30.44))
        start = end - pd.to_timedelta(days, unit="d")
        foo = lines[start < lines.index]
        for k in foo.keys():
            if foo[k].abs().sum() <= 0:
                foo = foo.drop(k, 1)
        title = "{} months".format(monthsDuration)
        plot = foo.plot.line(title=title, grid=True)
        plot.axes.set_ylabel("Euro")
        plot.axes.set_xlabel("Time")
        plot.legend(
            loc='center left',
            bbox_to_anchor=(1.0, 0.5)
        )
        """plot.get_figure().savefig(
            "{}.pdf".format(d), 
            bbox_inches = "tight"
        )"""
        plt.savefig(pdf, format='pdf', bbox_inches = "tight")
        plt.close()
    
    cake = lines.iloc[-1].round(2)
    for k in cake.keys():
        if cake[k] <= 0:
            cake = cake.drop(k, 0)
    cake = cake.drop("Investitionswert", 0)
    cake = cake.drop("Investitionssumme", 0)
    cake /= cake.sum()
    plot = cake.plot.pie(y="", autopct='%1.1f%%')
    plot.axes.set_ylabel("")
    plot.legend(
        loc='center left',
        bbox_to_anchor=(1.5, 0.5)
    )
    """.get_figure().savefig(
        "ratio.pdf", 
        bbox_inches = "tight",
        dpi=100
    )"""
    plt.savefig(pdf, format='pdf', bbox_inches = "tight")
    plt.close()
