import numpy as np
from shiny import ui,render
from shiny.types import ImgData
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import ast
#colorbling friendly plotting colours

data = load_breast_cancer(as_frame = True)
labels = data['target'].to_numpy()

def calc_MSE(dat, pred): 
    return np.mean((dat-pred)**2)

rng = np.random.default_rng(seed = 42)
true = (np.arange(10)/5)**3 + 2 + rng.standard_normal(10)*0.25
m1 = (np.arange(10)/5)**2 + 2
m2 = (np.arange(10)/5)**3+2
MSE1 = calc_MSE(true, m1)
MSE2 = calc_MSE(true, m2)
 



def dataset_gen(N_train, N_test): 
    #we "model" the relationship between time studying to points on exam
    rng = np.random.default_rng(seed = 42)
    x_range = np.linspace(0, 120, N_train+N_test)
    gt = -0.006*( x_range-120)**2+90
    noise = rng.standard_normal(N_train+N_test)*10
    noisy_data = gt + noise
    shuffled_ind = rng.permutation(np.arange(0, N_train+N_test))
    sort_train = np.argsort(shuffled_ind[:N_train])
    train_data = [x_range[shuffled_ind[:N_train]][sort_train], noisy_data[shuffled_ind[:N_train]][sort_train]]
    sort_test = np.argsort(shuffled_ind[:N_test])
    test_data = [x_range[shuffled_ind[N_train:]][sort_test], noisy_data[shuffled_ind[N_train:]][sort_test]]
    return train_data, test_data, [x_range, gt, noisy_data]

def simple_regression(deg, N_train , N_test):

    train_data, test_data, ground_truth = dataset_gen(N_train, N_test)

    reg = LinearRegression()
    poly_reg = Pipeline([("polynomial_features",PolynomialFeatures(degree = deg, include_bias = True)),
    ("linear_regression", reg),])
    poly_reg.fit(train_data[0].reshape(-1,1), train_data[1])

    predictions_train = poly_reg.predict(train_data[0].reshape(-1,1))

    predictions_test = poly_reg.predict(test_data[0].reshape(-1,1))

    return train_data, predictions_train, test_data, predictions_test, ground_truth


def gen_data_clustering(N1, N2, N3): 
    #1st dimension is coughing frequency
    #2nd dimension is the average activity level per week
    mean_smoker = np.array([10,4])
    sigma_smoker = np.array([5,2])
    mean_athletes = np.array([2,10])
    sigma_athletes = np.array([2, 1])
    mean_normal = np.array([5,6])
    sigma_normal = np.array([0.5, 3])

    rng = np.random.default_rng(seed = 42)
    dat_smokers = mean_smoker + sigma_smoker * rng.standard_normal(size = (N1, 2))
    dat_athletes = mean_athletes + sigma_athletes * rng.standard_normal(size = (N2,2))
    dat_normal = mean_normal + sigma_normal * rng.standard_normal(size = (N3, 2))

    dat_complete = np.vstack([dat_smokers, dat_athletes, dat_normal])
    dat_normalised = (dat_complete - np.mean(dat_complete, axis = 0))/np.std(dat_complete, axis = 0)
    
    #normalising the datasets: 
    dat_smokers = (dat_smokers - np.mean(dat_complete, axis = 0))/np.std(dat_complete, axis = 0)
    dat_athletes = (dat_athletes - np.mean(dat_complete, axis = 0))/np.std(dat_complete, axis = 0)
    dat_normal = (dat_normal - np.mean(dat_complete, axis = 0))/np.std(dat_complete, axis = 0)

    means_complete = np.vstack([np.mean(dat_smokers, axis = 0) , np.mean(dat_athletes, axis = 0), np.mean(dat_normal, axis = 0)])
    std_complete = np.vstack([np.std(dat_smokers, axis = 0) , np.std(dat_athletes, axis = 0), np.std(dat_normal, axis = 0)])
    

    return dat_complete, dat_normalised, dat_smokers, dat_athletes, dat_normal, means_complete, std_complete

def train_clf_bcw(layers = (30,10)): 

    data_points = data['data'].to_numpy()
    


    x_train, x_test, y_train, y_test = train_test_split(data_points, labels, test_size = 0.2, random_state= 42)

    #we have 30 features , so we increase the size and than decrease the size again
    clf = MLPClassifier(hidden_layer_sizes = layers, random_state=42, max_iter = 500 ).fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    acc = clf.score(x_test, y_test)
    #we add a small number to the denominator to make sure we do not divide by 0
    alpha_error = np.sum(y_pred[np.nonzero(y_test == 0)] == 1)/(np.sum(y_pred == 1)+1e-10)
    beta_error = np.sum(y_pred[np.nonzero(y_test == 1)] == 0)/(np.sum(y_pred == 0)+1e-10) 

    
    scores= [[round(1-alpha_error,2), round(beta_error,2)],
            [round(alpha_error,2), round(1-beta_error,2)]]


    return round(acc,2), scores


def make_table(data, row_labels, col_labels, row_super=None, col_super=None):
    ncols = len(col_labels)
    rows = []

    # Column supertitle row
    if col_super:
        rows.append(ui.tags.tr(ui.tags.th(""),ui.tags.th({"colspan": ncols}, col_super[0]),))
        rows.append(ui.tags.tr(ui.tags.th(""),ui.tags.th({"colspan": ncols}, col_super[1]),))
    # Column labels row
    rows.append(ui.tags.tr(ui.tags.th(row_super or ""),*[ui.tags.th(cl) for cl in col_labels],))

    # Data rows
    for r, row_label in enumerate(row_labels):
        rows.append(ui.tags.tr(ui.tags.th(row_label),*[ui.tags.td(data[r][c]) for c in range(ncols)],))

    return ui.tags.table({"class": "table table-bordered text-center align-middle"},*rows,)


#i simply contruct the page in here and than return it
#"lineare Regression, k-means, Klassifikation"



simple_AI_page = ui.page_fluid(
    ui.head_content(
        # Configure MathJax before loading it
        ui.tags.script("""
            window.MathJax = {
                tex: {
                    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
                },
                svg: { fontCache: 'global' }
            };
        """),
        ui.tags.script(src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js")
    ),
    ui.navset_card_tab(
        ui.nav_panel("Regression", 
        "Regression ist die einfachste Form eines 'KI'-Algorithmus. Dieser versucht basierend auf den Trainingsdaten die grundlegende Beziehung zu identifizieren. ",
        "Im Grunde versuchen wir eine Funktion zu finden, die minimal von allen erhobenen Datenpunkten entfernt ist. Im folgenden Beispiel sehen sie 2 mögliche Funktionen (orange, grün), die versuchen die Messdaten (blau) zu approximieren. Welche Funktion ist besser geeignet? ",
        ui.output_plot("reg_explanation"),
        ui.p("Um eine Funktion systematisch zu finden und damit eine Regression zu berechnen, optimierieren wird eine Metrik die 'mean-square error' (MSE) heißt. Der MSE misst den Abstand zwischen der Funktion ($\hat{y}$) und den Datenpunkten ($y$). Die optimierte Regressionsfunktion ist die Funktion, welche den kleinsten MSE hat."),
        ui.p(f"Ein $MSE = 0$ bedeuted, dass alle Datenpunkte auf der optimierten Funktion liegen. Funktion 1 hat ein MSE von $MSE(F1,y) =${MSE1:.2f} und Funktion 2 hat ein MSE von $MSE(F2,y) = ${MSE2:.2f}. Entspricht dies Ihrer Vermutung? ", ui.br(),"Die optimierte Regressionsfunktion wird oft auch als das 'Model' der Daten bezeichnet."),ui.br(), 
        "(Regressions-)Modelle werden durch die Anzahl ihrer Parameter charakterisiert. Je mehr Parameter ein Modell hat, desto komplexer die zugrundelegende Beziehungen. Die Anzahl der Parameter einer Regression lässt sich intuitiv als die Anzahl der Terme der Funktion verstehen: ", ui.br(), 
        "Funktion 1 hat 3 Parameter, da sie eine quadratische Funktion ist: $$\hat{y} = a x^0 + b x^1 + c x^2.$$" , ui.br(),
        "Funktion 2 hat 4 Parameter. Sie ist eine kubische Funktion: $$\hat{y} = a x^0+ b x^1 + c x^2 + d x^3.$$ ", 
        ui.p("Der erste Term $a x^0$ wird oft vereinfacht geschrieben als $a$, da $x^0 = 1$ für alle $x$. Da der Term eine Konstante ist, wird sie auch als 'Offset' bezeichnet. Eine Funktion mit 5 Parameter würde uns erlauben einen Term mit der Potenz 4 im Modell zu addieren. Eine Visualisering, wie sich die Funktionen mit der Anzahl der Parametern verändert sehen sie hier:"), 
        ui.output_plot("param_explanation"),
        "Im folgenden versuchen wir die Ergebnisse des folgenden (fiktiven) Beispiel mit einer Regression zu beschreiben. Es wurde die Testleistung von Probanden abhängig von der Zeit, die sie zum Lernen hatten gemessen. Die Verbesserung (im 2. Testzeitpunk im Vergleich zum 1. Testzeitpunkt) abhängig von der Lernzeit ist unten visualisiert (Figur: Gesamtdaten). Bei sämtlichen Optimierungsalgorithmus zählt es zum Standardverfahren die Gesamtdaten in ein Trainings- und Testset zu unterteilen. Die Figur 'Trainings-/Testset' zeigt eine solche Aufteilung. Aufgrund der Aufteilung berechnet man auch einen Trainings-MSE und einen Test-MSE. Warum könnte dies wichtig sein?", 
        ui.output_plot("vis_data"),
        "Wie vorhin erwähnt werden die Regressionsfunktion anhand der Anzahl der Parameter definiert. Daher müssen Sie vorher entscheiden, wie viele Parameter die Regression berücksichtigt. Mit dem Slider lassen sich die Anzahl der Parameter verändern. Sind mehr Parameter immer vorteilhaft? Wie verändert sich der Trainigs- und Test-MSE abhängig von der Parameteranzahl?",
        "in der Regression einstellen.",
        ui.div(ui.div(ui.input_slider("fl1", "Anzahl Parameter", min = 1, max=20, value = 5), style = "text-align:center;"),style = "display:flex; justify-content:center;"),
        ui.output_plot("regression_plot"),
        "Frage: Würden Sie zustimmen, dass je kleiner der MSE ist, desto besser ist das Modell?",ui.br(),
        "Frage: Wie unterscheidet sich das Modellverhalten zwischen den Trainingsset und den Testset?",ui.br(),ui.br(),
        "Da es sich hier um einen fiktiven Datensatz handelt, kennen wir die 'Ground Truth' und können die Funktion visualisieren, die den Gesamtdaten zugrunde liegt (Toggler 'Ground Truth'). Wie würden sie ein 'gutes' Modell charakterisieren abhängig von den MSE(s) und der Parameteranzahl?",
        ui.input_switch("show_gt", "Ground Truth", False),), 

        ui.nav_panel("Clustering", ui.HTML("<p>Clustering-Algorithmen bezeichnen Algorithmen, die die Gesamtdatenmenge in Gruppen einzuteilen. Ein wichtiger Unterschied zur Klassifizierung ist, dass die Datenmenge <b>keine</b> keine Label enthält. Daher zählt Clustering zu den <i> unsupervised Algorithmen </i>.</p> "),
        ui.HTML("<p>Anhand der <b>Distanz</b> zwischen den einzelnen Datenpunkten, werden sie einer der n Gruppen zugeteilt. Wie bei der Regression ist die Anzahl der Cluster/Gruppen ein <b>Hyperparameter</b>, welcher vorher festgelegt werden muss. Clustering kann vereinfacht zusammengefasst werden: <i> Datenpunkte, die nah beiander liegen, sind in dem selben Cluster. Datenpunkte, die weit entfernt voneinander sind, sind in unterschiedlichen Clustern.</i></p>"),
        "Die Abbildung visulisiert dieses Prinzip:",
        ui.output_plot("explanation_clustering"),
        ui.HTML("<p> Um die Auswirkungen der <b> Hyperparameterwahl</b> zu explorieren, beschäftigen wir uns im Folgenden mit einem (fiktiven) Datensatz zum Zusammenhang von Husten- und Sportfrequenz. Probanden haben über 4 Wochen dokumentiert, wie häufig sie Sport machen und wie häufig sie Hustenanfälle haben. Die Daten sind Frequnzen, die die durschnittliche Häufigkeit pro Woche angeben. Fällt Ihnen bei den Datenpunkten etwas auf?</p>"), 
        ui.output_plot("vis_clustering_data"),
        ui.input_switch("show_normed_dat", "Normalisiert", True),
        ui.HTML("<p> Der Datensatz ist normalisiert und es gibt negative Frequenzen. Normalisieren der Daten ist ein Standardverfahren, damit alle Merkmale in einem Wertebereich liegen, der optimal ist, um zu optimieren. Um die Daten zu normaliseren, berechnen wir den Mittelwert $\mu = \\frac{1}{n}\sum_i{x_i}$ und die Standardabweichung $\sigma = \\frac{1}{n} \sum_i{(x_i - \mu)^2}$ und sklaieren alle Datenpunkte $x_i$ mit den Werten: $$ x_{normalised} = \\frac{x_i - \mu}{\sigma}$$ Daraus folgt, dass $\mu_{normalised} = 0$ und $\sigma_{normalised} = 1$. Mit dem Toggler können Sie zwischen der Visualisierung der Rohdaten und dem normalisierten Datensatz wechseln. </p>"),
        ui.HTML("<p>Mit den normalisierten Daten führen wir ein Clustering durch. Der Algorithmus wird $K$-Means Clustering genannt. $K$ bezeichnet hier auch die Anzahl der Cluster, die benutzt wird. Mit dem Slider können Sie $K$ verändern. </p>"),
        
        ui.div(ui.div(ui.input_slider("clusters", "Anzahl Cluster", min = 2, max=10, value = 5), style = "text-align:center;"),style = "display:flex; justify-content:center;"),
        ui.output_plot("vis_kmean_cluster"),
        ui.br(),
        "Was fällt Ihnen auf? Wie viele Cluster vermuten Sie sind in den Daten wirklich vorhanden?",ui.br(), ui.br(),
        "Da es sich hier wieder um einen fiktiven Datensatz handelt, wissen wir wie viele Cluster verwendet wurden um das Datenset zu generieren. Mit dem Toggle, können Sie sich die zugrundelegende Clustergrenzen anzeigen lassen. ",
        ui.input_switch("show_gt_clustering", "Ground Truth", False),
        ui.HTML("<p>Dadurch das Clustering keine Labels verwendet und diese generell unbekannt sind, gibt es kein <i>korrektes</i> Clustering. In unseren Beispiel sehen wir, dass der große Überlapp zwischen Gruppe 1 und 3 dazu führt, dass der K-Mean Algorithmus Cluster berechnet, die weniger Überlapp haben. Können Sie erklären warum dies so ist?</p>")
        ),

        ui.nav_panel("Klassifikation", ui.HTML("<p>Im Gegensatz zu Clustering, benötigen Klassifikations-Algorithmen Labels, um zu predizieren welcher Datenpunk zu welcher Klasse gehört. Daher zählen Klassifizierungs-Algorithmen zu den <b>supervised</b> KI-Techniken. Die Funktionsweise kann vereinfacht beschrieben werden als: <i>Klassifizierungsalgorithmen optimieren Funktionen, die den (hochdimensionalen) Merkmalsraum so unterteilen, dass die höchste Anzahl an Datenpunkte korrekt klassifiziert werden.</i></p>"), 
        ui.HTML("<p>Dies bedeuted wir berechnen die <b>Accuracy</b> hinsichtlich des Klassenlabels und optimieren unser System basierend auf den Fehlern. Daher müssen wir auch wieder unsere Gesamtdatenmenge in ein Trainings- und Testset unterteilen. Das Trainingsset werden während der Optimierung verwendet, um das System zu trainieren und die Testdaten dienen der objektiven Bestimmung der Güte des trainierten Algorithmus.</p>"), 
        ui.br(), 
        ui.HTML("<p>Im folgenden benutzen wir den fiktiven Datensatz des Clusterings, um die Funktionsweise der Klassifizierung zu visualisieren. Allerdings haben unsere Daten ein weiteres Merkmal neben der Husten- und Sportfrequenz: die <b>Gruppe</b>. Die Entscheidungsfunktion des Algorithmus ist dargestellt durch die eingefärbte Fläche im Merkmalsraum. Alle Punkte, die im blauen Bereich sind, werden als Gruppe 1 zugehörig klassifiziert.</p>"), 
        ui.output_plot("toy_data_3D_static"), 
        ui.HTML("<p>Wie auch zuvor besitzen Klassifizierung <b> Hyperparameter</b>, die wir zuvor festlegen müssen. In diesem Fall ist es die Anzahl der Parameter. Eine nähere Erläuterung finden Sie weiter unten im 30-dimensionalen Klassifizierungsproblem. Die Auswahl ermöglicht Ihnen die Parameter im oberen Beispiel zu varriieren. Was fällt Ihnen auf? Gibt es Punkte, die immer falsch klassifiziert werden? Woran könnte dies liegen? </p>"), 
        ui.input_select("layers", "Parameter:", ["2","10","50","100","500"] ), 
        ui.output_plot("toy_data_3D"),
        ui.HTML("<p>Echte Datensätze sind häufig <b>hochdimensional</b>. Das heißt, es werden mehrere Merkmale pro Datenpunkt erfasst. Zudem sind die grundlenden Beziehung zwischen den Merkmalen und Gruppen unbekannt. KI (und Maschinelles Lernen) kann uns helfen die komplexen Beziehungen zwischen den Merkmalen und Klassenlabel zu approximieren. Im Folgenden beschäftigen wir uns mit dem <i>Breast Cancer Wisconsin</i> Datensatz. </p>"), ui.br(), 
        ui.HTML("<p>Der Datensatz erfasst die Messungen von 569 Tumorschnitten für die jeweils 30 Merkmale errechnet wurden, wie z.B. der durchschnittliche Radius oder die durchschnittliche Textur. Die Gutartigkeit der Gewebeschnitte wurde bestimmt und dient als Klassenlabel: 'malignant' and 'benign'. Die Tabelle zeigt Ihnen alle Merkmale der ersten 5 Proben an."), 
        ui.output_data_frame("class_data"),
        "Um einen besseren Überblick über die Struktur und Beziehung zwischen den erhobenn Merkmalen zu erlangen, können Sie unten 2 Merkmale auswählen und sich diese darstellen lassen. Die Farbe zeigt ihnen die Klassenzugehörigkeit an.",
        ui.input_select("first_feature", "1. Merkmal:", choices = data['data'].keys().tolist()),
        ui.input_select("second_feature", "2. Merkmal:", choices = data['data'].keys().tolist()),
        ui.input_select("third_feature", "3. Merkmal:", choices = data['data'].keys().tolist()),
        ui.output_plot("class_plot_3D"),
        "Sehen Sie Merkmalskombinationen, bei denen sich die beiden Klassen klar von einander trennen lassen? Können Sie sich Funktionen vorstellen, die die Klassen von einander trennen können in 3D? Was bedeutet dies für uns, wenn wir alle 30 Merkmale berücksichtigen?", ui.br(),ui.br(),
        ui.HTML("<p> Da wir 2 Klassen haben, lassen sich die Prediktionen in 4 Kategoriern aufteilen und bewerten:</p><ul><li>Korrekt Positiv: Spezifität</li><li>Korrekt Negativ: Sensitivität</li><li>Falsch Positiv: $\\alpha$-Fehler</li><li>Falsch Negativ: $\\beta$-Fehler</></ul>"),
        ui.HTML("<p>Diese Begriffen sollten Ihnen aus der Statistik und Testtheorie bereits bekannt sein. Auf unseren Datensatz angewendet mit dem Ziel korrekt bösartige (malignant) Tumore zu identifizieren ergibt sich folgende Tabelle: </p>"),
        ui.output_ui("exp_table"),
        ui.HTML("Welche Werten würden Sie als akzeptabel empfinden?"), ui.br(),
        ui.HTML("<p>Im folgenden trainieren wir ein <b>Multi-Layer-Perceptron (MLP)</b> mit dem Datensatz, den wir zuvor wieder in ein Test- und ein Trainingsset aufgeteilt haben. Ein MLP besteht aus mehreren Schichten von Neuronen. Die Anzahl der Schichten, wie auch die Anzahl der Neurone und deren Verbindungen in jeder Schicht wird zusammen als <b>Architektur</b> des Netzwerkes bezeichnet. Die Architektur mit all seinen Eigenschaften ist wieder ein <b>Hyperparameter</b>, den wir selber apriori (also vorher) bestimmen müssen. Um Ihnen eine bessere Vorstellung von einem MLP zu geben, zeigt die Architektur eines (100,15)-MLP. Die zahlen bedeuten, dass das MLP 100 Neurone in der 2. Schicht und 15 Neurone in der 3. Schicht hat. Der Output sind 2 Neurone: Je eins per Klasse. </p>"),
        ui.br(), ui.br(),
        ui.output_image("architecture",fill=True),
        ui.HTML("<p>Neurone sind als Kreise dargestellt. Die Abbildung zeigt nicht alle Gewichte des Netzwerkes. Jedes Neuron im Input ist mit jedem Neuron in der 1. Schicht verbunden. Jedes Neuron in der 1. Schicht ist mit jedem Neuron in der 2. Schicht verbunden usw.. Die Anzahl der <Parameter>, die in einem MLP trainiert (also optimiert) werden sind abhängig von der Anzahl der Neurone. Diese Parameter sind die Verbindungen/Gewichte zwischen den Neuronen, daher hat ein MLP wie oben $30 \cdot 100 \cdot 15$ Parameter.</p>"),
        ui.HTML("<p>Erinnern Sie sich an den Zusammenhang zwischen der Anzahl der Parameter und der Performance der Regression? Denken Sie der Zusammenhang hier ist auch zo gegeben? Woran merkt man, dass das trainierte Netzwerk zu groß ist? </p>"),
        ui.HTML("<p>Das folgende Menu erlaubt Ihnen verschiedene Architekturen auszuwählen und zu trainieren. Die Ergbenisse werden Ihnen in der Tabelle darunter angezeigt. Was fällt Ihnen auf? Welche Performance empfinden Sie als akzeptabel?"),
        ui.input_select("MLP_param", "Architektur:", ["(100,15,)","(100,5,)","(100,50,)","(50,15,)","(200,15,)"] ),
        ui.output_ui("acc_table"),ui.br(), ui.br(),
        )
        
    ))
 



#all render functions
def server_AI_examples(input): 
    import matplotlib.style as style
    style.use('seaborn-colorblind')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cm_bright = ListedColormap(["#0000FF","#FF0000"])

    style.use('seaborn-bright')
    bright = plt.rcParams['axes.prop_cycle'].by_key()['color']
    cm_bright3D = ListedColormap([bright[0],bright[1],bright[2]])

    style.use('seaborn-colorblind')
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    cm_3D = ListedColormap([palette[0],palette[1],palette[2] ])

    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    n_train = 20
    n_test = 20
    n_cluster = 20



    @render.plot
    def vis_data():
        train_dat, test_dat, ground_truth = dataset_gen(n_train, n_test)
        fig, ax = plt.subplots(1,2)
        ax[0].scatter(ground_truth[0], ground_truth[2], color = palette[0])
        ax[0].set_xlabel("Min. Lernen")
        ax[0].set_ylabel("% Verbesserung")
        ax[0].set_title("Gesamtdaten")
        ax[1].scatter(train_dat[0], train_dat[1], color = palette[0], label = "Training")
        ax[1].scatter(test_dat[0], test_dat[1], color = palette[2], label = "Test")
        ax[1].set_xlabel("Min. Lernen")
        ax[1].set_ylabel("% Verbesserung")
        ax[1].legend(bbox_to_anchor=(1.0, 1.0), loc='upper left',  frameon = False)
        ax[1].set_title("Trainings-/Testdaten")


    @render.plot
    def regression_plot():

        dat_train, pred_train, dat_test, pred_test, real_data = simple_regression(input.fl1()-1, n_train, n_test)
        fig, ax = plt.subplots(1,2)
        ax[0].scatter(dat_train[0], dat_train[1], color = palette[0], label = 'Daten')
        ax[0].plot(dat_train[0], pred_train, color = palette[2], label = 'Model')
        if input.show_gt(): 
            ax[0].plot(real_data[0], real_data[1], color = 'k', label = 'Ground Truth')
        ax[0].set_title("Trainings-MSE: "+str(round(calc_MSE(dat_train[1], pred_train),2)))
        ax[1].scatter(dat_test[0], dat_test[1], color = palette[0], label = 'Daten')
        ax[1].scatter(dat_test[0], pred_test, color = palette[2], label = 'Model')
        if input.show_gt(): 
            ax[1].plot(real_data[0], real_data[1], color = 'k', label = 'Ground Truth')
        ax[1].set_title("Test-MSE: "+str(round(calc_MSE(dat_test[1], pred_test),2)))
        ax[0].set_xlabel("Min. Lernen")
        ax[0].set_ylabel("% Verbesserung")
        ax[0].legend(bbox_to_anchor=(1.0, 1.0), loc='upper left',  frameon = False)
        ax[1].set_xlabel("Min. Lernen")
        ax[1].set_ylabel("% Verbesserung")
        ax[1].legend(bbox_to_anchor=(1.0, 1.0), loc='upper left',  frameon = False)

    @render.plot
    def param_explanation(): 
        ex_dat = np.arange(-2,2, 0.01)
        x_range = np.arange(-20,20,.1)
        factors = [0.75* ex_dat **0,ex_dat**1,ex_dat**2,ex_dat**3-ex_dat,ex_dat**4-2*ex_dat**2,ex_dat**5 - 4*ex_dat**3 - ex_dat]
        for i in range(6): 
            if i > 0: 
                fun_normed = factors[i]/np.abs(np.min(factors[i]) - np.max(factors[i]))
                plt.plot(x_range, fun_normed, label = str(i+1)+' Parameter')
            else: 
                fun_normed = factors[i]
                plt.plot(x_range, fun_normed, label = 'Offset')
        plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left',  frameon = False)

    

    @render.plot
    def reg_explanation():
        with plt.xkcd():
            plt.scatter(np.arange(10),true, label = 'Daten')
            plt.plot(np.arange(10), m1,  color = palette[2], label = 'Funktion 1')
            plt.plot(np.arange(10), m2, color = palette[1], label = 'Funktion 2')
            plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left',  frameon = False)
        
    


    @render.plot
    def explanation_clustering():
        from matplotlib.patches import Ellipse
        rng = np.random.default_rng(seed = 42)
        group1 = np.array([2,1]) + np.array([0.5,0.5]) * rng.standard_normal(size = (25,2))
        group2 = np.array([1,2]) + np.array([0.5, 0.75]) * rng.standard_normal(size = (20,2))
        group3 = np.array([3,2]) + np.array([0.75,0.2]) * rng.standard_normal(size = (25,2))
        
        with plt.xkcd():
            fig, ax = plt.subplots(1,1)
            ax.scatter(group1[:,0], group1[:,1], color = palette[0])
            ax.add_artist(Ellipse((2,1),1.5,1.5, angle = 0, alpha = 0.1, color = palette[0]))
            ax.text(1, 0.75, "Cluster 2", fontsize = 20, color = palette[0], ha = 'center', va = 'center')
            ax.scatter(group2[:,0], group2[:,1], color = palette[1])
            ax.add_artist(Ellipse((1,2),1.5,2.25, angle = -25, alpha = 0.1, color = palette[1]))
            ax.text(1, 3, "Cluster 1", fontsize = 20, color = palette[1], ha = 'center', va = 'center')
            ax.scatter(group3[:,0], group3[:,1], color = palette[2])
            ax.add_artist(Ellipse((3,2),3,1, angle = 0, alpha = 0.1, color = palette[2]))
            ax.text(3, 2.5, "Cluster 3", fontsize = 20, color = palette[2], ha = 'center', va = 'center')
            ax.plot([1.26, 1.34], [2.382, 2.55], color = 'k')
            ax.text(1.45, 2.425, "$d_{in}$", fontsize = 16, color = 'k', ha = 'center', va = 'center', rotation = -13)
            ax.plot([1.28, 2.2], [2.315, 2.05], color = 'grey')
            ax.text(1.7, 2.1, "$d_{between}$", fontsize = 16, color = 'grey', ha = 'center', va = 'center', rotation = -13 )
            ax.text(4,1, "$d_{in} < d_{between}$", ha = 'center', va = 'center')
        

    @render.plot
    def vis_clustering_data(): 
        comp, normed, _, _, _ ,_,_= gen_data_clustering(n_cluster, n_cluster, n_cluster)
        fig, ax = plt.subplots(1,1) 
        if input.show_normed_dat():
            #not sure if flipping is better or putting it on top
            ax.scatter(normed[:,0], normed[:,1])
        #if not  input.show_normed_dat():
        else: 
            ax.scatter(comp[:,0], comp[:,1])
        ax.set_xlabel("Hustenfrequenz")
        ax.set_ylabel("Sportfrequenz")

    @render.plot
    def vis_kmean_cluster():
        from matplotlib.patches import Ellipse
        nnormed, comp, smoke, ath, norm,means, stds = gen_data_clustering(n_cluster, n_cluster, n_cluster)
        kmeans_cluster = KMeans(n_clusters = input.clusters(), random_state = 42, n_init = "auto").fit(comp)
        if input.show_gt_clustering(): 
            fig, ax = plt.subplots(1,2)
            ax[1].scatter(smoke[:,0], smoke[:,1], color = palette[0], label = 'Gruppe 1')
            ax[1].scatter(ath[:,0], ath[:,1], color = palette[1], label = 'Gruppe 2')
            ax[1].scatter(norm[:,0], norm[:,1], color = palette[2], label = 'Gruppe 3') 
            for i in range(means.shape[0]):
                ax[1].scatter(means[i,0], means[i,1], color = palette[i], marker = '*', s = 40, edgecolors = 'k')
            ax[1].legend(bbox_to_anchor=(1.0, 1.0), loc='upper left',  frameon = False)
            ax[1].add_artist(Ellipse((means[0]),stds[0,0]*5,stds[0,1]*5, angle = 0, alpha = 0.1, color = palette[0]))
            ax[1].add_artist(Ellipse((means[1]),stds[1,0]*5,stds[1,1]*5, angle = 25, alpha = 0.1, color = palette[1]))
            ax[1].add_artist(Ellipse((means[2]),stds[2,0]*5,stds[2,1]*5, angle = 180, alpha = 0.1, color = palette[2]))
            ax[1].set_xlabel("Hustenfrequenz")
            ax[1].set_ylabel("Sportfrequenz")
            ax[1].set_title("Ground Truth")
            scatter = ax[0].scatter(comp[:,0], comp[:,1], c = kmeans_cluster.labels_)
            ax[0].scatter(kmeans_cluster.cluster_centers_[:,0],kmeans_cluster.cluster_centers_[:,1], c = np.arange(input.clusters(), dtype = int),  marker = '*', s = 40, edgecolors = 'k' )
            ax[0].legend(*scatter.legend_elements(),bbox_to_anchor=(1.0, 1.0), loc='upper left',  frameon = False)
            ax[0].set_xlabel("Hustenfrequenz")
            ax[0].set_ylabel("Sportfrequenz")
            ax[0].set_title("Model")

        else: 
            fig, ax = plt.subplots(1,1)
            scatter = ax.scatter(comp[:,0], comp[:,1], c = kmeans_cluster.labels_)
            ax.scatter(kmeans_cluster.cluster_centers_[:,0],kmeans_cluster.cluster_centers_[:,1], c = np.arange(input.clusters(), dtype = int),  marker = '*', s = 40, edgecolors = 'k' )
            ax.legend(*scatter.legend_elements(),bbox_to_anchor=(1.0, 1.0), loc='upper left',  frameon = False)
            ax.set_xlabel("Hustenfrequenz")
            ax.set_ylabel("Sportfrequenz")
            
    @render.plot
    def toy_data_3D_static():

        nndata, data, _, _, _, _, _ = gen_data_clustering(n_cluster, n_cluster, n_cluster)

        labels = np.zeros([n_cluster*3])
        labels[n_cluster:n_cluster*2] = 1
        labels[n_cluster*2:] = 2

        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, random_state= 42)

        clf = MLPClassifier(hidden_layer_sizes = (10,), random_state=42, max_iter = 1000).fit(x_train, y_train)

        samples = 100
        x_decs, y_decs = np.meshgrid(np.linspace(data[:,0].min()-0.5, data[:,0].max()+0.5,samples), np.linspace(data[:,1].min()-0.5, data[:,1].max()+0.5,samples))
        decs = clf.predict_proba(np.column_stack([x_decs.ravel(), y_decs.ravel()]))
        decs0 = decs[:,0]
        decs1 = decs[:,1]
        decs2 = decs[:,2]
        #compile them into one map which indicates which class is most likely: 
        class_regime = np.zeros(decs[:,1].shape)
        for i in range(decs.shape[0]): 
            if decs0[i] > decs1[i]: 
                if decs0[i] > decs2[i]:
                    class_regime[i] = 0.0
                elif decs2[i] > decs1[i]:
                    class_regime[i] = 1.0
            elif decs1[i] > decs2[i]: 
                class_regime[i] = 0.5

        class_regime = class_regime.reshape(x_decs.shape)
        with plt.xkcd(): 
            fig, ax = plt.subplots(1,1)
            ax.contourf(x_decs, y_decs, class_regime, cmap = cm_3D, alpha = 0.75)
            scatter1 = ax.scatter(x_train[:,0],x_train[:,1], c = (y_train*0.5), cmap = cm_bright3D , edgecolors = 'k')
            handles = scatter1.legend_elements()[0]
            l1 = ax.legend(handles, ['Gruppe 1','Grupp 2','Gruppe 3'],title= 'Trainingsset', bbox_to_anchor=(1.0, 1.0), loc='upper left',  frameon = False)
            l1._legend_box.align = 'left'
            ax2 = ax.twinx()
            scatter2 = ax2.scatter(x_test[:,0],x_test[:,1], c = (y_test*0.5), cmap = cm_bright3D , edgecolors = 'k',marker = "*" )
            ax2.get_yaxis().set_visible(False)
            handles = scatter2.legend_elements()[0]
            l2 = ax2.legend(handles, ['Gruppe 1','Grupp 2','Gruppe 3'],title= 'Testset', bbox_to_anchor=(1.0,0.6), loc='upper left',  frameon = False)
            l2._legend_box.align = 'left'
            ax.annotate("Prediktion inkorrekt",  xytext=(0.5, 1.2), xy=(-.1, 1.65),arrowprops=dict(arrowstyle="->"), fontsize = 10)
            ax.annotate("Prediktion korrekt",  xytext=(1.5, -2), xy=(1.25, -1.55),arrowprops=dict(arrowstyle="->"), fontsize = 10)
            ax.annotate("Trainingsfehler",  xytext=(0.8, 0.2), xy=(-0.1, 0.55),arrowprops=dict(arrowstyle="->"), fontsize = 10)
            ax.set_xlabel("Hustenfrequenz")
            ax.set_ylabel("Sportfrequenz")

    @render.plot
    def toy_data_3D():

        nndata, data, _, _, _, _, _ = gen_data_clustering(n_cluster, n_cluster, n_cluster)

        labels = np.zeros([n_cluster*3])
        labels[n_cluster:n_cluster*2] = 1
        labels[n_cluster*2:] = 2

        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, random_state= 42)

        clf = MLPClassifier(hidden_layer_sizes = (np.int(input.layers()),), random_state=42, max_iter = 1000 ).fit(x_train, y_train)

        samples = 100
        x_decs, y_decs = np.meshgrid(np.linspace(data[:,0].min()-0.5, data[:,0].max()+0.5,samples), np.linspace(data[:,1].min()-0.5, data[:,1].max()+0.5,samples))
        decs = clf.predict_proba(np.column_stack([x_decs.ravel(), y_decs.ravel()]))
        decs0 = decs[:,0]
        decs1 = decs[:,1]
        decs2 = decs[:,2]
        #compile them into one map which indicates which class is most likely: 
        class_regime = np.zeros(decs[:,1].shape)
        for i in range(decs.shape[0]): 
            if decs0[i] > decs1[i]: 
                if decs0[i] > decs2[i]:
                    class_regime[i] = 0.0
                elif decs2[i] > decs1[i]:
                    class_regime[i] = 1.0
            elif decs1[i] > decs2[i]: 
                class_regime[i] = 0.5

        class_regime = class_regime.reshape(x_decs.shape)
        acc = clf.score(x_test, y_test)

        fig, ax = plt.subplots(1,1)
        ax.contourf(x_decs, y_decs, class_regime, cmap = cm_3D, alpha = 0.75)
        scatter1 = ax.scatter(x_train[:,0],x_train[:,1], c = (y_train*0.5), cmap = cm_bright3D , edgecolors = 'k')
        handles = scatter1.legend_elements()[0]
        l1 = ax.legend(handles, ['Gruppe 1','Grupp 2','Gruppe 3'],title= 'Trainingsset', bbox_to_anchor=(1.0, 1.0), loc='upper left',  frameon = False)
        l1._legend_box.align = 'left'
        ax2 = ax.twinx()
        scatter2 = ax2.scatter(x_test[:,0],x_test[:,1], c = (y_test*0.5), cmap = cm_bright3D , edgecolors = 'k',marker = "*" )
        ax2.get_yaxis().set_visible(False)
        handles = scatter2.legend_elements()[0]
        l2 = ax2.legend(handles, ['Gruppe 1','Grupp 2','Gruppe 3'],title= 'Testset', bbox_to_anchor=(1.0,0.6), loc='upper left',  frameon = False)
        l2._legend_box.align = 'left'
        ax.set_xlabel("Hustenfrequenz")
        ax.set_ylabel("Sportfrequenz")
        ax.set_title("Accuracy: "+str(acc)+"%")
 



    @render.data_frame
    def class_data(): 
        return render.DataGrid(data['data'].head(5))

    @render.image
    def architecture():
        from pathlib import Path
        dir = Path(__file__).resolve().parent
        img: ImgData = {"src": str(dir / "architektur.png")}
        return img


    @render.plot()
    def class_plot_3D():
        
        x_data = data['data'][input.first_feature()]
        y_data = data['data'][input.second_feature()]
        z_data = data['data'][input.third_feature()]
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        scatter = ax.scatter3D(x_data, y_data, z_data, c = labels, cmap = cm_bright)
        handles = scatter.legend_elements()[0]
        ax.legend(handles, ['malignant','benign',],bbox_to_anchor=(1.0, 1.0), loc='upper left',  frameon = False)
        ax.set_xlabel(input.first_feature())
        ax.set_ylabel(input.second_feature())
        ax.set_zlabel(input.third_feature())

    @render.ui
    def exp_table():
        row_labels = ["Malignant", "Benign"]
        col_labels = ["Malignant", "Benign"]
        tab_dat = [["Spezifität",ui.HTML("&beta;-Fehler")],[ui.HTML("&alpha;-Fehler"), "Sensitivität"]]
        return make_table(tab_dat,row_labels,col_labels,col_super=["","Tumor ist"],row_super="Diagnostiziert als",)


    @render.ui
    def acc_table():
        acc_dat, tab_dat =  train_clf_bcw(layers = ast.literal_eval(input.MLP_param()))
        row_labels = ["Malignant", "Benign"]
        col_labels = ["Malignant", "Benign"]
        return make_table(tab_dat,row_labels,col_labels,col_super=["Accuracy: "+str(acc_dat)+"%","Tumor ist"],row_super="Diagnostiziert als",)






