import EDA

class EDAReport:
    def __init__(self, df):
        self.df = df
    #Fully automated report for EDA and save it as pdf
    def report(self):
        """ fully automated report """
        #EDA
        eda = EDA(self.df)
        print("----------Info----------")
        eda.info()
        print()
        print()
        print("----------Unique categories----------")
        eda.unique_categories()
        print()
        print()
        print("----------Missing values----------")
        eda.isna()
        print()
        print()
        print("----------Duplicate values----------")
        eda.duplicate_value()
        print()
        print()
        print("----------Statical describe of numerical variables----------")
        eda.describe()
        print()
        print()
        print("----------Histogram----------")
        eda.hist()
        print()
        print()
        print("----------Distplot----------")
        eda.dist()
        print()
        print()
        print("----------Boxplot----------")
        eda.box()
        print()
        print()
        print("----------Counterplot----------")
        eda.counter()
        print()
        print()
        print("----------histplot----------")
        eda.hist_hue()
        print()
        print()
        print("----------violinplot----------")
        eda.violin()

    
        # Create a string to capture the printed output
        report_output = []

        # Redirect print statements to capture the output
        def capture_output(text):
            report_output.append(text)

        original_print = print
        print = capture_output

        # Call the methods to capture the output
        self.info()
        self.unique_categories()
        self.isna()
        # ... Call other methods ...

        # Restore the original print function
        print = original_print

        return "\n".join(report_output)

    def save_report_to_pdf(self, filename):
        pdf_content = self.report()

        c = canvas.Canvas(filename, pagesize=letter)
        y = 750

        for line in pdf_content.split("\n"):
            c.drawString(100, y, line)
            y -= 20

        c.save()
        print(f"PDF report saved as {filename}")


