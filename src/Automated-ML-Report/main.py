import EDAReport

class main:
    def __init__(self,df):
        self.df = df

    obj = EDAReport(self.df)
    obj.report()