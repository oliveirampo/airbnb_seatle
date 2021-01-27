import matplotlib.pyplot as plt
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go


def plot_box_data(df, attribute, unique_values, title):
    box_data = []

    for val in unique_values:
        trace = go.Box(
            x=df.loc[df[attribute] == val].price.tolist(),
            name=val)
        box_data.append(trace)

    box_layout = go.Layout(xaxis=dict(title='Listing Price'), title=title)
    box_fig = go.Figure(data=box_data, layout=box_layout)
    return box_fig


def main():
    df_listing_detailed = pd.read_csv('inp/listings_detailed.csv')
    df_listing_detailed['price'] = df_listing_detailed['price'].str.replace("[$, ]", "").astype("float")


if __name__ == "__main__":
    main()