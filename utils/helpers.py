import os 
import array

import plotly.express as px
import plotly.io as pio
from sklearn.decomposition import PCA

def apply_pca(data: array, dim=2):
    # Use pca to reduce dimensionality
    pca = PCA(n_components=dim)
    data = pca.fit_transform(data)
    return data

def create_plotly_scatter_plot(df, x_data, y_data, color_name=None, hover_name=None, width=800, height=600, save_path=None):
    # create scatter plot using plotly
    fig = px.scatter(df, x=x_data, y=y_data, color=color_name, hover_name=hover_name, width=width, height=height)
    if save_path is not None:
        print(f'Saving plot in path:\n {save_path}')
        # pio.write_image(fig, save_path)
        pio.write_html(fig, save_path)
    else:
        fig.show()


def delete_file(file_path):
    # delete existing files
    print('Deleting existing files!!!')
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
            print(f"Deleted {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")