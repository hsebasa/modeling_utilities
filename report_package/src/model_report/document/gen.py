import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO

class HTMLReport:
    @staticmethod
    def heading(text, level=1):
        return f"<h{level}>{text}</h{level}>"

    @staticmethod
    def paragraph(text):
        return f"<p>{text}</p>"

    @staticmethod
    def table(data, border=1):
        table_html = f"<table border='{border}'><tr>"
        table_html += "".join(f"<th>{col}</th>" for col in data[0])
        table_html += "</tr>"
        
        for row in data[1:]:
            table_html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        
        table_html += "</table>"
        return table_html

    @staticmethod
    def dataframe(df, **kwargs):
        return df.to_html(**kwargs)

    @staticmethod
    def list_html(items, ordered=False):
        tag = "ol" if ordered else "ul"
        return f"<{tag}>" + "".join(f"<li>{item}</li>" for item in items) + f"</{tag}>"

    @staticmethod
    def figure(fig=None):
        if fig is None:
            fig = plt.gcf()
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        return f"<img src='data:image/png;base64,{encoded}'/>"


class HTMLReport:
    def __init__(self, title="Report"):
        self.title = title
        self.elements = []
    
    def add_heading(self, text, level=1):
        self.elements.append(f"<h{level}>{text}</h{level}>")
    
    def add_paragraph(self, text):
        self.elements.append(f"<p>{text}</p>")
    
    def add_table(self, data, border=1, title=None):
        table_html = f"<table border='{border}'><tr>"
        table_html += "".join(f"<th>{col}</th>" for col in data[0])
        table_html += "</tr>"
        
        for row in data[1:]:
            table_html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        
        table_html += "</table>"
        wrapped_html = f"<div>{f'<b>{title}</b><br>' if title else ''}{table_html}</div>"
        self.elements.append(wrapped_html)
    
    def add_dataframe(self, df, title=None, **kwargs):
        df_html = df.to_html(**kwargs)
        wrapped_html = f"<div>{f'<b>{title}</b><br>' if title else ''}{df_html}</div>"
        self.elements.append(wrapped_html)
    
    def add_list(self, items, ordered=False, title=None):
        tag = "ol" if ordered else "ul"
        list_html = f"<{tag}>" + "".join(f"<li>{item}</li>" for item in items) + f"</{tag}>"
        wrapped_html = f"<div>{f'<b>{title}</b><br>' if title else ''}{list_html}</div>"
        self.elements.append(wrapped_html)
    
    def add_figure(self, fig=None, title=None):
        if fig is None:
            fig = plt.gcf()
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        img_html = f"<img src='data:image/png;base64,{encoded}'/>"
        wrapped_html = f"<div>{f'<b>{title}</b><br>' if title else ''}{img_html}</div>"
        self.elements.append(wrapped_html)
    
    def render(self):
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.title}</title>
        </head>
        <body>
            {''.join(self.elements)}
        </body>
        </html>
        """
        return html_content