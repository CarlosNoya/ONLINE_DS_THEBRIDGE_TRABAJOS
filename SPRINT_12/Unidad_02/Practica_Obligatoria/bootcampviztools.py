import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


def describe_df(df):
    """
    Función para describir un DataFrame de pandas proporcionando información sobre el tipo de datos,
    valores faltantes, valores únicos y cardinalidad.

    Params:
        df: DataFrame de pandas.

    Returns:
        DataFrame con la información recopilada sobre el DataFrame de entrada.
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError("El argumento 'df' debe ser un DataFrame de pandas válido.")

    # Creamons un diccionario para almacenar la información
    data = {
        'DATA_TYPE': df.dtypes,
        'MISSINGS(%)': df.isnull().mean() * 100,
        'UNIQUE': df.nunique(),
        'CARD(%)': round(df.nunique() / len(df) * 100, 3)
    }

    # Creamos un nuevo DataFrame con la información recopilada, usamos 'transpose' para cambiar
    # las filas por columnas.
    estudiantes_df = pd.DataFrame(data).transpose()

    return estudiantes_df


def tipifica_variable_plus(df, umbral_categoria=10, umbral_continua=30.0, mostrar_card=True):
    """
    Función para tipificar variables como binaria, categórica, numérica continua y numérica discreta.
    
    Params:
        df (pd.DataFrame): DataFrame de pandas.
        umbral_categoria (int): Valor entero que define el umbral de la cardinalidad para variables categóricas.
        umbral_continua (float): Valor flotante que define el umbral de la cardinalidad para variables numéricas continuas.
        motrar_card (bool): Si es True, incluye la cardinalidad y el porcentaje de cardinalidad. True por defecto. 
    
    Returns:
        DataFrame con las columnas (variables), la tipificación sugerida de cada una.     
        y el tipo real detectado por pandas. Si `motrar_card` es True, también incluye las columnas 
        "CARD" (cardinalidad absoluta) y "%_CARD" (porcentaje de cardinalidad relativa).
        Incluye también el % de valores "missings" de cada variable
    """
    
    # Validación del DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("El argumento 'df' debe ser un DataFrame de pandas válido.")
    
    # Validación de los umbrales
    if not isinstance(umbral_categoria, int) or umbral_categoria <= 0:
        raise ValueError("El 'umbral_categoria' debe ser un número entero mayor que 0.")
    
    if not isinstance(umbral_continua, (float, int)) or umbral_continua <= 0:
        raise ValueError("El 'umbral_continua' debe ser un número float mayor que 0.")
    
    # DataFrame inicial con cardinalidad y tipificación sugerida
    df_card = pd.DataFrame({
        "Dtype_real": df.dtypes.astype(str),
        "Missings_%": round((df.isnull().mean() * 100),2),
        "CARD_valores_unicos": df.nunique(),
        "CARD_%": round((df.nunique() / len(df) * 100),2),
        "Dtype_sugerido": ""
    })
    
    # Tipo Binaria
    df_card.loc[df_card["CARD_valores_unicos"] == 2, "Dtype_sugerido"] = "Binaria"
    
    # Tipo Categórica
    df_card.loc[(df_card["CARD_valores_unicos"] < umbral_categoria) & (df_card["Dtype_sugerido"] == ""), "Dtype_sugerido"] = "Categórica"
    
    # Tipo Numérica Continua
    df_card.loc[(df_card["CARD_valores_unicos"] >= umbral_categoria) & (df_card["CARD_%"] >= umbral_continua), "Dtype_sugerido"] = "Numérica Continua"
    
    # Tipo Numérica Discreta
    df_card.loc[(df_card["CARD_valores_unicos"] >= umbral_categoria) & (df_card["CARD_%"] < umbral_continua), "Dtype_sugerido"] = "Numérica Discreta"

    # Selección y renombrado de columnas
    df_card = df_card.reset_index().rename(columns={"index": "Variable"})
    
    if mostrar_card == False:
        print("Umbral de 'CARD_valores_unicos' para considerarla 'Categórica': ", umbral_categoria)
        print("Umbral de 'CARD_%' para considerarla 'Numérica Continua': ", umbral_continua)
        return df_card[["Variable", "Missings_%", "Dtype_real", "Dtype_sugerido"]]
        
    else:
        print("Umbral de 'CARD_valores_unicos' para considerarla 'Categórica': ", umbral_categoria)
        print("Umbral de 'CARD_%' para considerarla 'Numérica Continua': ", umbral_continua)
        return df_card[["Variable","Missings_%", "Dtype_real", "CARD_valores_unicos", "CARD_%", "Dtype_sugerido"]]


def pinta_distribucion_categoricas(df, columnas_categoricas, relativa=False, mostrar_valores=False):
    num_columnas = len(columnas_categoricas)
    num_filas = (num_columnas // 2) + (num_columnas % 2)

    fig, axes = plt.subplots(num_filas, 2, figsize=(15, 5 * num_filas))
    axes = axes.flatten() 

    for i, col in enumerate(columnas_categoricas):
        ax = axes[i]
        if relativa:
            total = df[col].value_counts().sum()
            serie = df[col].value_counts().apply(lambda x: x / total)
            sns.barplot(x=serie.index, y=serie, ax=ax, palette='viridis', hue = serie.index, legend = False)
            ax.set_ylabel('Frecuencia Relativa')
        else:
            serie = df[col].value_counts()
            sns.barplot(x=serie.index, y=serie, ax=ax, palette='viridis', hue = serie.index, legend = False)
            ax.set_ylabel('Frecuencia')

        ax.set_title(f'Distribución de {col}')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)

        if mostrar_valores:
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    for j in range(i + 1, num_filas * 2):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def plot_categorical_relationship_fin(df, cat_col1, cat_col2, relative_freq=False, show_values=False, size_group = 5):
    # Prepara los datos
    count_data = df.groupby([cat_col1, cat_col2]).size().reset_index(name='count')
    total_counts = df[cat_col1].value_counts()
    
    # Convierte a frecuencias relativas si se solicita
    if relative_freq:
        count_data['count'] = count_data.apply(lambda x: x['count'] / total_counts[x[cat_col1]], axis=1)

    # Si hay más de size_group categorías en cat_col1, las divide en grupos de size_group
    unique_categories = df[cat_col1].unique()
    if len(unique_categories) > size_group:
        num_plots = int(np.ceil(len(unique_categories) / size_group))

        for i in range(num_plots):
            # Selecciona un subconjunto de categorías para cada gráfico
            categories_subset = unique_categories[i * size_group:(i + 1) * size_group]
            data_subset = count_data[count_data[cat_col1].isin(categories_subset)]

            # Crea el gráfico
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x=cat_col1, y='count', hue=cat_col2, data=data_subset, order=categories_subset)

            # Añade títulos y etiquetas
            plt.title(f'Relación entre {cat_col1} y {cat_col2} - Grupo {i + 1}')
            plt.xlabel(cat_col1)
            plt.ylabel('Frecuencia' if relative_freq else 'Conteo')
            plt.xticks(rotation=45)

            # Mostrar valores en el gráfico
            if show_values:
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', fontsize=10, color='black', xytext=(0, size_group),
                                textcoords='offset points')

            # Muestra el gráfico
            plt.show()
    else:
        # Crea el gráfico para menos de size_group categorías
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=cat_col1, y='count', hue=cat_col2, data=count_data)

        # Añade títulos y etiquetas
        plt.title(f'Relación entre {cat_col1} y {cat_col2}')
        plt.xlabel(cat_col1)
        plt.ylabel('Frecuencia' if relative_freq else 'Conteo')
        plt.xticks(rotation=45)

        # Mostrar valores en el gráfico
        if show_values:
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=10, color='black', xytext=(0, size_group),
                            textcoords='offset points')

        # Muestra el gráfico
        plt.show()


def plot_categorical_numerical_relationship(df, categorical_col, numerical_col, show_values=False, measure='mean'):
    # Calcula la medida de tendencia central (mean o median)
    if measure == 'median':
        grouped_data = df.groupby(categorical_col)[numerical_col].median()
    else:
        # Por defecto, usa la media
        grouped_data = df.groupby(categorical_col)[numerical_col].mean()

    # Ordena los valores
    grouped_data = grouped_data.sort_values(ascending=False)

    # Si hay más de 5 categorías, las divide en grupos de 5
    if grouped_data.shape[0] > 5:
        unique_categories = grouped_data.index.unique()
        num_plots = int(np.ceil(len(unique_categories) / 5))

        for i in range(num_plots):
            # Selecciona un subconjunto de categorías para cada gráfico
            categories_subset = unique_categories[i * 5:(i + 1) * 5]
            data_subset = grouped_data.loc[categories_subset]

            # Crea el gráfico
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x=data_subset.index, y=data_subset.values)

            # Añade títulos y etiquetas
            plt.title(f'Relación entre {categorical_col} y {numerical_col} - Grupo {i + 1}')
            plt.xlabel(categorical_col)
            plt.ylabel(f'{measure.capitalize()} de {numerical_col}')
            plt.xticks(rotation=45)

            # Mostrar valores en el gráfico
            if show_values:
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                                textcoords='offset points')

            # Muestra el gráfico
            plt.show()
    else:
        # Crea el gráfico para menos de 5 categorías
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=grouped_data.index, y=grouped_data.values)

        # Añade títulos y etiquetas
        plt.title(f'Relación entre {categorical_col} y {numerical_col}')
        plt.xlabel(categorical_col)
        plt.ylabel(f'{measure.capitalize()} de {numerical_col}')
        plt.xticks(rotation=45)

        # Mostrar valores en el gráfico
        if show_values:
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                            textcoords='offset points')

        # Muestra el gráfico
        plt.show()


def plot_combined_graphs(df, columns, whisker_width=1.5, bins = None):
    num_cols = len(columns)
    if num_cols:
        
        fig, axes = plt.subplots(num_cols, 2, figsize=(12, 5 * num_cols))
        print(axes.shape)

        for i, column in enumerate(columns):
            if df[column].dtype in ['int64', 'float64']:
                # Histograma y KDE
                sns.histplot(df[column], kde=True, ax=axes[i,0] if num_cols > 1 else axes[0], bins= "auto" if not bins else bins)
                if num_cols > 1:
                    axes[i,0].set_title(f'Histograma y KDE de {column}')
                else:
                    axes[0].set_title(f'Histograma y KDE de {column}')

                # Boxplot
                sns.boxplot(x=df[column], ax=axes[i,1] if num_cols > 1 else axes[1], whis=whisker_width)
                if num_cols > 1:
                    axes[i,1].set_title(f'Boxplot de {column}')
                else:
                    axes[1].set_title(f'Boxplot de {column}')

        plt.tight_layout()
        plt.show()

def plot_grouped_boxplots(df, cat_col, num_col):
    unique_cats = df[cat_col].unique()
    num_cats = len(unique_cats)
    group_size = 5

    for i in range(0, num_cats, group_size):
        subset_cats = unique_cats[i:i+group_size]
        subset_df = df[df[cat_col].isin(subset_cats)]
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=cat_col, y=num_col, data=subset_df)
        plt.title(f'Boxplots of {num_col} for {cat_col} (Group {i//group_size + 1})')
        plt.xticks(rotation=45)
        plt.show()



def plot_grouped_histograms(df, cat_col, num_col, group_size):
    unique_cats = df[cat_col].unique()
    num_cats = len(unique_cats)

    for i in range(0, num_cats, group_size):
        subset_cats = unique_cats[i:i+group_size]
        subset_df = df[df[cat_col].isin(subset_cats)]
        
        plt.figure(figsize=(10, 6))
        for cat in subset_cats:
            sns.histplot(subset_df[subset_df[cat_col] == cat][num_col], kde=True, label=str(cat))
        
        plt.title(f'Histograms of {num_col} for {cat_col} (Group {i//group_size + 1})')
        plt.xlabel(num_col)
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()



def grafico_dispersion_con_correlacion(df, columna_x, columna_y, tamano_puntos=50, mostrar_correlacion=False):
    """
    Crea un diagrama de dispersión entre dos columnas y opcionalmente muestra la correlación.

    Args:
    df (pandas.DataFrame): DataFrame que contiene los datos.
    columna_x (str): Nombre de la columna para el eje X.
    columna_y (str): Nombre de la columna para el eje Y.
    tamano_puntos (int, opcional): Tamaño de los puntos en el gráfico. Por defecto es 50.
    mostrar_correlacion (bool, opcional): Si es True, muestra la correlación en el gráfico. Por defecto es False.
    """

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=columna_x, y=columna_y, s=tamano_puntos)

    if mostrar_correlacion:
        correlacion = df[[columna_x, columna_y]].corr().iloc[0, 1]
        plt.title(f'Diagrama de Dispersión con Correlación: {correlacion:.2f}')
    else:
        plt.title('Diagrama de Dispersión')

    plt.xlabel(columna_x)
    plt.ylabel(columna_y)
    plt.grid(True)
    plt.show()


def bubble_plot(df, col_x, col_y, col_size, scale = 1000):
    """
    Crea un scatter plot usando dos columnas para los ejes X e Y,
    y una tercera columna para determinar el tamaño de los puntos.

    Args:
    df (pd.DataFrame): DataFrame de pandas.
    col_x (str): Nombre de la columna para el eje X.
    col_y (str): Nombre de la columna para el eje Y.
    col_size (str): Nombre de la columna para determinar el tamaño de los puntos.
    """

    # Asegúrate de que los valores de tamaño sean positivos
    sizes = (df[col_size] - df[col_size].min() + 1)/scale

    plt.scatter(df[col_x], df[col_y], s=sizes)
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.title(f'Burbujas de {col_x} vs {col_y} con Tamaño basado en {col_size}')
    plt.show()


def get_features_num_regression(df, target_col, umbral_corr, pvalue=None):
    """
    Función para seleccionar características basadas en la correlación con la variable objetivo.

    Params:
        df: DataFrame de pandas.
        target_col: Nombre de la columna objetivo en el DataFrame.
        umbral_corr: Valor flotante que define el umbral de la correlación para seleccionar características.
        pvalue: Valor flotante opcional que define el umbral de significancia para filtrar características basadas en el valor p.

    Returns:
        Lista de características que cumplen con los criterios de selección.
    """
    # Comprobaciones de los argumentos de entrada
    if not isinstance(df, pd.DataFrame):
        # comprueba que el primer argumento sea un DataFrame
        print("El primer argumento debe ser un DataFrame.")
        return None
    if target_col not in df.columns:
        # comprueba que la columna target exista en el DataFrame
        print(f"La columna '{target_col}' no existe en el DataFrame.")
        return None
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        # comprueba que la columna target sea numérica
        print(f"La columna '{target_col}' no es numérica.")
        return None
    if not (0 <= umbral_corr <= 1):
        # comprueba que el umbral de correlación esté entre 0 y 1
        print("El umbral de correlación debe estar entre 0 y 1.")
        return None
    if pvalue is not None and not (0 <= pvalue <= 1):
        # comprueba que el valor de pvalue esté entre 0 y 1
        print("El valor de pvalue debe estar entre 0 y 1.")
        return None

        # Calcular la correlación
    # 'abs' calcula el valor absoluto de las correlaciones.
    corr = df.corr()[target_col].abs()
    features = corr[corr > umbral_corr].index.tolist()
    # Eliminar la variable target de la lista de features, porque su valor de correlación es 1.
    features.remove(target_col)

    # Filtrar por pvalue si es necesario
    if pvalue is not None:
        significant_features = []
        for feature in features:
            # colocamos el guión bajo '_,' para indicar que no nos interesa el primer valor
            _, p_val = stats.pearsonr(df[feature], df[target_col])
            if p_val < pvalue:
                significant_features.append(feature)
        features = significant_features

    return features


def plot_features_num_regression(df, target_col="", columns=[], umbral_corr=0, pvalue=None):
    """
    Función para analizar correlaciones numéricas y graficar pairplots.

    Params:
        df (pd.DataFrame): DataFrame de entrada.
        target_col (str): Columna objetivo para calcular correlaciones.
        columns (list[str]): Columnas a evaluar (si está vacío, selecciona numéricas). Por defecto [].
        umbral_corr (float): Umbral absoluto de correlación. Por defecto 0.
        pvalue (float): Nivel de significancia para el p-valor (opcional). Por defecto None.

    Returns:
        list[str]: Columnas seleccionadas que cumplen con los criterios.
    """

    # Validación del DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            "El argumento 'df' debe ser un DataFrame de pandas válido.")

    # Validación de que 'target_col' sea obligatorio
    if not target_col:
        raise ValueError(
            "Debes proporcionar un 'target_col' válido para calcular correlaciones.")

    # Verificamos si 'target_col' es una columna válida en el dataframe
    if target_col and target_col not in df.columns:
        raise ValueError(f"La columna indicada como 'target_col': {target_col} no está en el DataFrame.")

    # Verificamos si la columna 'target_col' es numérica
    if target_col and not pd.api.types.is_numeric_dtype(df[target_col]):
        raise ValueError(f"La columna indicada como 'target_col': {target_col} no es numérica.")

    # Si 'columns' está vacío, usamos todas las columnas numéricas excepto 'target_col'
    if not columns:
        columns = df.select_dtypes(include=['number']).columns.tolist()
        if target_col in columns:
            columns.remove(target_col)

    # sino, es decir, si 'columns' no está vacío, validamos que las columnas existan y sean numéricas
    else:
        invalid_cols = [col for col in columns if col not in df.columns]
        if invalid_cols:
            raise ValueError(
                f"Las siguientes columnas no están en el DataFrame: {invalid_cols}")

        non_numeric_cols = [
            col for col in columns if not pd.api.types.is_numeric_dtype(df[col])]
        if non_numeric_cols:
            raise ValueError(f"Las siguientes columnas no son numéricas: {non_numeric_cols}")

    # Validación del umbral de correlacion 'umbral_corr'
    if not isinstance(umbral_corr, (int, float)) or not 0 <= umbral_corr <= 1:
        raise ValueError(
            "El argumento 'umbral_corr' debe ser un número entre 0 y 1.")

    # Validación del valor 'P-Value'
    if pvalue is not None:
        if not isinstance(pvalue, (int, float)) or not 0 <= pvalue <= 1:
            raise ValueError(
                "El argumento 'pvalue' debe ser un número entre 0 y 1, o 'None'.")

    # CORRELACION
    # Correlación absoluta de las columnas vs al target
    corr = np.abs(df.corr()[target_col]).sort_values(ascending=False)

    # Columnas que superan el umbral de correlacion, excluyendo a la columna 'target_col'
    selected_columns = corr[(corr > umbral_corr) & (
        corr.index != target_col)].index.tolist()

    # Si se proporciona un p-value, verificamos significancia estadística
    if pvalue is not None:
        significant_columns = []  # Lista para guardar las columnas significativas
        for col in selected_columns:
            corr, p_val = stats.pearsonr(df[col], df[target_col])
            if p_val <= pvalue:
                significant_columns.append(col)
        selected_columns = significant_columns  # Actualizamos con las significativas

    # Validación de los resultados de la correlación
    if not selected_columns:
        print("No se encontraron columnas que cumplan con los criterios de correlación y p-valor.")
        return None

    # PAIRPLOTS
    # Incluir siempre target_col en cada gráfico
    columns_to_plot = [target_col] + selected_columns

    # Dividir en grupos de máximo 5 columnas (incluyendo target_col)
    num_groups = (len(columns_to_plot) - 1) // 4 + \
        1  # Grupos de 4 columnas + target

    for i in range(num_groups):
        # Seleccionar un grupo de columnas
        subset = columns_to_plot[:1] + columns_to_plot[1 + i*4:1 + (i+1)*4]

        # Generar el pairplot
        sns.pairplot(df[subset], diag_kind="kde", corner=True)
        plt.show()

    return selected_columns


def check_normality(data):
    stat, p = stats.shapiro(data)
    return p > 0.01

def check_homoscedasticity(*groups):
    stat, p = stats.levene(*groups)
    return p > 0.01

def get_features_cat_regression(df, target_col, pvalue=0.05):
    """
    Función para obtener las características categóricas significativas en un modelo de regresión lineal.
    Params:
                df: dataframe de pandas
                target_col: columna objetivo del dataframe
                pvalue: p-valor para el test de significancia
    Returns:
                Lista con las características categóricas significativas
        """
    # Verificamos si el dataframe es válido
    if not isinstance(df, pd.DataFrame):
        print("El argumento 'df' no es un dataframe válido.")
        return None
    if not (0 <= pvalue <= 1):
        # comprueba que el valor de pvalue esté entre 0 y 1
        print("El valor de pvalue debe estar entre 0 y 1.")
        return None
    # Verificamos si 'target_col' es una columna válida en el dataframe
    if target_col not in df.columns:
        print(f"La columna '{target_col}' no está en el dataframe.")
        return None
    # Verificamos si la columna 'target_col' es numérica
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"La columna '{target_col}' no es una columna numérica.")
        return None
    # Identificar las columnas categóricas del dataframe
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not cat_columns:
        print("No se encontraron características categóricas en el dataframe.")
        return None
    # Lista para almacenar las columnas categóricas que superan el pvalor
    significant_cat_features = []
    for cat_col in cat_columns:
        # Si la columna categórica tiene más de un nivel (para que sea válida para el test)
        if df[cat_col].nunique() > 1:
            try:
                groups = [df[target_col][df[cat_col] == level].dropna() for level in df[cat_col].unique()]
                if all(len(g) >= 2 for g in groups):
                    # Comprobamos normalidad y homocedasticidad
                    all_data = np.concatenate(groups)
                    is_normal = check_normality(all_data)
                    is_homoscedastic = check_homoscedasticity(*groups)

                    if is_normal and is_homoscedastic:
                        print(f"La distribución de {target_col} es normal y homocedástica.")
                        if len(groups) == 2:
                            t_stat, p_val = stats.ttest_ind(groups[0], groups[1])
                            print(f"Student t (p_value): {p_val}")
                        else:
                            f_val, p_val = stats.f_oneway(*groups)
                            print(f"Oneway ANOVA (p_value): {p_val}")
                    else:
                        print(f"La distribución de {target_col} NO es normal o homocedástica.")
                        if len(groups) == 2:
                            u_stat, p_val = stats.mannwhitneyu(groups[0], groups[1])
                            print(f"MannWhitney U (p_value): {p_val}")
                        else:
                            h_stat, p_val = stats.kruskal(*groups)
                            print(f"Kruskal (p_value): {p_val}")

                    # Comprobamos si el p-valor es menor que el p-valor especificado
                    if p_val < pvalue:
                        significant_cat_features.append(cat_col)
            except Exception as e:
                print(f"Error al procesar la columna {cat_col}: {str(e)}")
                continue
    # Si encontramos columnas significativas, las devolvemos
    if significant_cat_features:
        return significant_cat_features
    else:
        print("\nNo se encontraron características categóricas significativas.")
        return None

def plot_features_cat_regression(df, target_col="", columns=[], pvalue=0.05, with_individual_plot=False):
    """
    Función para graficar histogramas agrupados de variables categóricas significativas.
    Params:
        df: dataframe de pandas
        target_col: columna objetivo del dataframe (variable numérica)
        columns: lista de columnas categóricas a evaluar (si está vacía, se usan todas las columnas categóricas)
        pvalue: p-valor para el test de significancia
        with_individual_plot: si es True, genera un gráfico individual por cada categoría
    Returns:
        Lista de columnas que cumplen con los criterios de significancia
    """
    # Verificamos si el dataframe es válido
    if not isinstance(df, pd.DataFrame):
        print("El argumento 'df' no es un dataframe válido.")
        return None
    if target_col and target_col not in df.columns:
        print(f"La columna '{target_col}' no está en el dataframe.")
        return None
    if not (0 <= pvalue <= 1):
        # comprueba que el valor de pvalue esté entre 0 y 1
        print("El valor de pvalue debe estar entre 0 y 1.")
        return None
    if target_col and not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"La columna '{target_col}' no es una columna numérica.")
        return None
    # Identificar las columnas categóricas del dataframe
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not cat_columns:
        print("No se encontraron características categóricas en el dataframe.")
        return None
    # Si 'columns' está vacío, usamos todas las columnas categóricas
    if not columns:
        columns = cat_columns
    # Lista para almacenar las columnas significativas
    significant_cat_features = []
    # Verificamos la significancia de las columnas categóricas con respecto al 'target_col'
    for cat_col in columns:
        if cat_col not in cat_columns:
            print(f"La columna '{cat_col}' no es categórica o no existe en el dataframe.")
            continue
        if df[cat_col].nunique() > 1:
            try:
                groups = [df[target_col][df[cat_col] == level].dropna() for level in df[cat_col].unique()]
                if all(len(g) >= 2 for g in groups):
                    # Comprobamos normalidad y homocedasticidad
                    all_data = np.concatenate(groups)
                    is_normal = check_normality(all_data)
                    is_homoscedastic = check_homoscedasticity(*groups)

                    if is_normal and is_homoscedastic:
                        print(f"La distribución de {target_col} es normal y homocedástica.")
                        if len(groups) == 2:
                            t_stat, p_val = stats.ttest_ind(groups[0], groups[1])
                            print(f"Student t (p_value): {p_val}")
                        else:
                            f_val, p_val = stats.f_oneway(*groups)
                            print(f"Oneway ANOVA (p_value): {p_val}")
                    else:
                        print(f"La distribución de {target_col} NO es normal o homocedástica.")
                        if len(groups) == 2:
                            u_stat, p_val = stats.mannwhitneyu(groups[0], groups[1])
                            print(f"MannWhitney U (p_value): {p_val}")
                        else:
                            h_stat, p_val = stats.kruskal(*groups)
                            print(f"Kruskal (p_value): {p_val}")

                    # Comprobamos si el p-valor es menor que el p-valor especificado
                    if p_val < pvalue:
                        significant_cat_features.append(cat_col)
            except Exception as e:
                print(f"Error al procesar la columna {cat_col}: {str(e)}")
                continue
    # Si no hay columnas significativas
    if not significant_cat_features:
        print("\nNo se encontraron características categóricas significativas.")
        return None
    # Graficar histogramas agrupados para las columnas significativas
    for cat_col in significant_cat_features:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=target_col, hue=cat_col,
                     kde=True, multiple="stack", bins=30)
        plt.title(f"Distribución de {target_col} por {cat_col}")
        plt.xlabel(target_col)
        plt.ylabel("Frecuencia")
        plt.show()
        # Si 'with_individual_plot' es True, graficar histogramas individuales por cada categoría
        if with_individual_plot:
            for level in df[cat_col].unique():
                plt.figure(figsize=(10, 6))
                sns.histplot(df[df[cat_col] == level],
                             x=target_col, kde=True, bins=30)
                plt.title(f"Distribución de {target_col} para {cat_col} = {level}")
                plt.xlabel(target_col)
                plt.ylabel("Frecuencia")
                plt.show()
    return significant_cat_features