import numpy as np
from shiny import ui,render
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
    poly_reg = Pipeline([("polynomial_features",PolynomialFeatures(degree = deg, include_bias = False)),
    ("linear_regression", reg),])
    poly_reg.fit(train_data[0].reshape(-1,1), train_data[1])

    predictions_train = poly_reg.predict(train_data[0].reshape(-1,1))

    predictions_test = poly_reg.predict(test_data[0].reshape(-1,1))

    return train_data, predictions_train, test_data, predictions_test, ground_truth


def gen_data_clustering(N1, N2, N3): 
    #1st dimension is coughing frequency [normalised, 0 for normal]
    #2nd dimension is the average activity level per week [normalised, 0 for normal]
    mean_smoker = np.array([0.5,0])
    sigma_smoker = np.array([0.7,0.5])
    mean_athletes = np.array([-0.5,0.5])
    sigma_athletes = np.array([0.5, 0.2])
    mean_normal = np.array([0,0])
    sigma_normal = np.array([0.5, 0.5])

    rng = np.random.default_rng(seed = 42)
    dat_smokers = mean_smoker + sigma_smoker * rng.standard_normal(size = (N1, 2))
    dat_athletes = mean_athletes + sigma_athletes * rng.standard_normal(size = (N2,2))
    dat_normal = mean_normal + sigma_normal * rng.standard_normal(size = (N3, 2))

    dat_complete = np.vstack([dat_smokers, dat_athletes, dat_normal])
    means_complete = np.vstack([mean_smoker, mean_athletes, mean_normal])
    return dat_complete, dat_smokers, dat_athletes, dat_normal, means_complete

def train_clf_bcw(layers = (30,10), features = None): 
    if features: 
        data_points = data['data'][[features[0], features[1]]].to_numpy()
    else: 
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

    #get a linescore for 2 dims
    if features: 
        samples = 100
        x_decs, y_decs = np.meshgrid(np.linspace(data_points[:,0].min()-0.5, data_points[:,0].max()+0.5,samples), np.linspace(data_points[:,1].min()-0.5, data_points[:,1].max()+0.5,samples))
        decs = clf.predict_proba(np.column_stack([x_decs.ravel(), y_decs.ravel()]))[:,1]
        decs = decs.reshape(x_decs.shape)
    
        return data_points, decs, x_decs, y_decs
    
    else: 
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
        "Funktion 1 hat 2 Parameter, da sie eine quadratische Funktion ist: $$\hat{y} = a x^0 + b x^1 + c x^2.$$" , ui.br(),
        "Funktion 2 hat 3 Parameter. Sie ist eine kubische Funktion: $$\hat{y} = a x^0+ b x^1 + c x^2 + d x^3.$$ ", 
        ui.p("Der erste Term $a x^0$ wird oft vereinfacht geschrieben als $a$, da $x^0$ eine constante Funktion mit Wert 1 ist. Da der Term eine Konstante ist, wird sie 'Bias' genannt. Eine Funktion mit 4 Parameter würde uns erlauben einen Term mit der Potenz 4 im Modell zu adiren. Eine Visualisering, wie sich die Funktionen mit der Anzahl der Parametern verändert sehen sie hier:"), 
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

        ui.nav_panel("Gruppierung", "Gruppierung oder auch clustering (im Folgenden) sind Algorithmen, die versuchen in Gruppen einzuteilen.",
        "Clustering ist eine unsupervised Methode. Das heißt, der Algorithmus hat keinen Label oder Klassen zur Verfügung. Die Gruppierung",
        "erfolgt allinig auf den Merkmalen der Datenpunkt. Oft wird die Nähe der Datenpunkte als Indikator benutzt, dass sie zu der selben Gruppen",
        "gehören. In der Regel gilt: Daten der selben Gruppe liegen nahe zueinander, während Datenpunkte anderer Gruppen weit entfernt sind.", ui.br(),
        "Die Abbildung visulisiert dieses Prinzip:",
        ui.output_plot("explanation_clustering"),
        "Wir befassen uns mit den folgenden (fiktiven) Datensatz in dem die Husten- und Sportfrequenz gemessen wurden. Beide Maße sind normalisiert.", 
        ui.output_plot("vis_clustering_data"),
        "Den Algorithmus den wir nutzen um Gruppierungen zu finden heißt K-Means Clustering. Er benutzt das oben beschriebene Prinzip.",
        "Im Folgenden können Sie über den Slider Auswählen wie viele Cluster Sie annehmen in den Daten vorhanden sind.",ui.br(),
        "Frage: Was fällt Ihnen auf? Wie viele Cluster vermuten Sie sind in den Daten wirklich vorhanden?",ui.br(), ui.br(),
        ui.div(ui.div(ui.input_slider("clusters", "Anzahl Cluster", min = 2, max=10, value = 5), style = "text-align:center;"),style = "display:flex; justify-content:center;"),
        ui.output_plot("vis_kmean_cluster"),
        "Mit den Button unten können Sie sich die wirklich Datenverteilung anzeigen lassen.",
        ui.input_switch("show_gt_clustering", "Ground Truth", False),
        ),

        ui.nav_panel("Klassifikation", "Klassifikations-Algorithmen versuchen Datensätzen mit vielen Eigenschaften in die angegebenen Klassen zu gruppieren.", 
        "Konkret bedeutet dies, dass wir für die Trainingsdaten die Klassenzugehörigkeit kennen. Die Testdaten sind allerdings Datenpunkte bei denen wir nicht wissen", 
        "welcher Klasse der Datenpunkt angehört. Das Ziel ist es den neuen Daten die korrekte Klasse zuzuweisen.", ui.br(), 
        "Mathematisch versucht ein Klassifizierungsalgorithmus Funktionen zu finden, die die Grenzen zwischen den Klassen optimal beschreiben.",
        "Diese werden unter der folgenden Bedingungen optimiert: ",
        "Die maximale Anzahl an Trainingspunkten sollen richtig klassifiziert werden.", ui.br(),
        "Die meisten Algorithmen implementieren weitere Bedingungen, die den Parameterraum einschränken.",
        "Klassifizierungsalgorithmen können folgenderweise visualisiert werden: ", 
        ui.output_plot("toy_data_3D"),
        "Die unterschiedlich gefärbten Bereiche zeigen an, welche Klasse für Datenpunkte an diesen Stellen prediziert würde.", 
        "Sie können die Anzahl der Parameter mit dem folgenden Slider einstellen.", 
        ui.input_select("layers", "Parameter:", ["2","10","50","100","500"] ), 
        "Was fällt Ihnen auf?", ui.br(), ui.br(),
        "Im Folgenden beschäftigen wir uns mit dem 'breast cancer Wisconsin' Datensatz. ", ui.br(), 
        "Dieser beinhaltet 569 Messpunkte mit jeweils 30 Eigenschaften. Die Datenpunkte sind in 2 Klassen eingeteilt: 'malignant' and 'benign'. Die Tabelle zeigt ihnen die ersten 5 Messpunkte an.", 
        ui.output_data_frame("class_data"),
        "Um einen besseren Überblick über die Struktur und Beziehung zwischen den erhobenn Merkmalen zu erlangen, können Sie unten 2 Merkmale auswählen und sich diese darstellen lassen. Die Farbe zeigt ihnen die Klassenzugehörigkeit an.",
        ui.input_select("first_feature", "1. Merkmal:", choices = data['data'].keys().tolist()),
        ui.input_select("second_feature", "2. Merkmal:", choices = data['data'].keys().tolist()),
        ui.output_plot("class_plot_2D"),
        "Sehen Sie Merkmalskombinationen, bei denen sich die beiden Klassen klar von einander trennen lassen?", ui.br(),
        "Um neue Daten automatisch zu Klassifizieren trainieren wir ein Multi-Layer-Perceptron (MLP) mit unseren hochdimensionalen Daten. Das trainierte MLP kann dann für 'neue' Messungen predizieren, ob die Biopsie von einem gut (benign) oder bösartigen (malignant) Tumor stammt.",
        "Durch die 2 Klassen ist dies ein binäres Klassifizierungsproblem. Das heißt wir können einfach ausrechnen wie oft der Algorithmus Tumore fehldiagnostiziert. Probieren Sie aus, wie sich die Performance ändert, wenn Sie die Architektur des MLPs ändern.", ui.br(), 
        "Info: Die Anzahl der Parameter in einem MLP ist equivalent zu dem Produkt der Neuronen pro Schicht. Das heißt unser 'Standard'-(100,15,)-MLP  hat 1500 Parameter (100 Neuronen in der 1. Schicht und 15 Neuronen in der 2. Schicht).",
        ui.input_select("MLP_param", "Architektur:", ["(100,15,)","(100,5,)","(100,50,)","(50,15,)","(200,15,)"] ),
        ui.output_ui("acc_table"),ui.br(), ui.br(),
        "Da es nicht möglich ist die Entscheidungsfunktion mit hinsicht auf alle 30 Merkmale darzustellen, benuzten wir im folgenden nur 2 Merkmale. Können sie durch die drop-down selection selber auswählen.",
        "Die Visualisierung zeigt die Wahrscheinlichkeit mit der ein Datenpunkt als benign klassifiziert wird. Je tiefer der Rotton desto sicherer ist der Algorithmus, dass es sich um einen benign Tumor handelt. Je tieder der Blauton, desto wahrscheinlicher ist der Datenpunkt malignant.",
        "Weiß indiziert eine Klassenzugehörigkeit von 0.5 bzw. 50%. Dies bedeutet, dass der Algorithmus sich nicht sicher ist, welcher Klasse der Datenpunkt angehört.", 
        "Unten können Sie wieder auswählen, welche 2 Merkmale vom MLP berücksichtigt werden.", ui.br(), ui.br(),
        "Wie würden Sie die Entscheidungsfunktion des MLPs beschreiben?", 
        ui.input_select("first2D_feature", "1. Merkmal:", choices = data['data'].keys().tolist()),
        ui.input_select("second2D_feature", "2. Merkmal:", choices = data['data'].keys().tolist()),
        ui.output_plot("clf_2D"))
        
    ))
 



#all render functions
def server_AI_examples(input): 
    import matplotlib.style as style
    style.use('seaborn-v0_8-colorblind')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cm_bright = ListedColormap(["#0000FF","#FF0000"])

    style.use('seaborn-v0_8-bright')
    bright = plt.rcParams['axes.prop_cycle'].by_key()['color']
    cm_bright3D = ListedColormap([bright[0],bright[1],bright[2]])

    style.use('seaborn-v0_8-colorblind')
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

        dat_train, pred_train, dat_test, pred_test, real_data = simple_regression(input.fl1(), n_train, n_test)
        fig, ax = plt.subplots(1,2)
        ax[0].scatter(dat_train[0], dat_train[1], color = palette[0], label = 'Daten')
        ax[0].plot(dat_train[0], pred_train, color = palette[2], label = 'Model')
        if input.show_gt(): 
            ax[0].plot(real_data[0], real_data[1], color = 'k', label = 'Ground Truth')
        ax[0].set_title("Trainings-MSE: "+str(calc_MSE(dat_train[1], pred_train)))
        ax[1].scatter(dat_test[0], dat_test[1], color = palette[0], label = 'Daten')
        ax[1].scatter(dat_test[0], pred_test, color = palette[2], label = 'Model')
        if input.show_gt(): 
            ax[1].plot(real_data[0], real_data[1], color = 'k', label = 'Ground Truth')
        ax[1].set_title("Test-MSE: "+str(calc_MSE(dat_test[1], pred_test)))
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
                plt.plot(x_range, fun_normed, label = str(i)+' Parameter')
            else: 
                fun_normed = factors[i]
                plt.plot(x_range, fun_normed, label = 'Bias')
        plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left',  frameon = False)

    

    @render.plot
    def reg_explanation():
        plt.scatter(np.arange(10),true, label = 'Daten')
        plt.plot(np.arange(10), m1,  color = palette[2], label = 'Funktion 1')
        plt.plot(np.arange(10), m2, color = palette[1], label = 'Funktion 2')
        plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left',  frameon = False)
        
    


    @render.plot
    def explanation_clustering():
        plt.scatter(np.arange(10), np.arange(10))

    @render.plot
    def vis_clustering_data(): 
        comp, _, _, _ ,_= gen_data_clustering(n_cluster, n_cluster, n_cluster)
        fig, ax = plt.subplots(1,1)
        ax.scatter(comp[:,0], comp[:,1])
        ax.set_xlabel("Hustenfrequenz")
        ax.set_ylabel("Sportfrequenz")

    @render.plot
    def vis_kmean_cluster():

        comp, smoke, ath, norm,means = gen_data_clustering(n_cluster, n_cluster, n_cluster)
        kmeans_cluster = KMeans(n_clusters = input.clusters(), random_state = 42, n_init = "auto").fit(comp)
        if input.show_gt_clustering(): 
            fig, ax = plt.subplots(1,2)
            ax[1].scatter(smoke[:,0], smoke[:,1], color = palette[0], label = 'Bauarbeiter')
            ax[1].scatter(ath[:,0], ath[:,1], color = palette[1], label = 'Athleten')
            ax[1].scatter(norm[:,0], norm[:,1], color = palette[2], label = 'Lehrer') 
            for i in range(means.shape[0]):
                ax[1].scatter(means[i,0], means[i,1], color = palette[i], marker = '*', s = 40, edgecolors = 'k')
            ax[1].legend(bbox_to_anchor=(1.0, 1.0), loc='upper left',  frameon = False)
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
    def toy_data_3D():

        data, _, _, _, means = gen_data_clustering(n_cluster, n_cluster, n_cluster)

        labels = np.zeros([n_cluster*3])
        labels[n_cluster:n_cluster*2] = 1
        labels[n_cluster*2:] = 2

        clf = MLPClassifier(hidden_layer_sizes = (np.int(input.layers()),), random_state=42, max_iter = 1000 ).fit(data, labels)

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

        fig, ax = plt.subplots(1,1)
        ax.contourf(x_decs, y_decs, class_regime, cmap = cm_3D, alpha = 0.75)
        scatter = ax.scatter(data[:,0],data[:,1], c = (labels*0.5), cmap = cm_bright3D , edgecolors = 'k')
        scatter = ax.scatter(means[:,0],means[:,1], c = [0,0.5,1], cmap = cm_bright3D , edgecolors = 'k', marker = '*', s = 40)
        handles = scatter.legend_elements()[0]
        ax.legend(handles, ['Bauarbeiter','Athleten','Lehrer'],bbox_to_anchor=(1.0, 1.0), loc='upper left',  frameon = False)
        ax.set_xlabel("Hustenfrequenz")
        ax.set_ylabel("Sportfrequenz")
 



    @render.data_frame
    def class_data(): 
        return render.DataGrid(data['data'].head(5))

    @render.plot()
    def class_plot_2D():
        x_data = data['data'][input.first_feature()]
        y_data = data['data'][input.second_feature()]
        fig, ax = plt.subplots(1,1)
        scatter = ax.scatter(x_data, y_data, c = labels, cmap = cm_bright)
        handles = scatter.legend_elements()[0]
        ax.legend(handles, ['malignant','benign',],bbox_to_anchor=(1.0, 1.0), loc='upper left',  frameon = False)
        ax.set_xlabel(input.first_feature())
        ax.set_ylabel(input.second_feature())


    @render.ui
    def acc_table():
        acc_dat, tab_dat =  train_clf_bcw(layers = ast.literal_eval(input.MLP_param()))
        row_labels = ["Malignant", "Benign"]
        col_labels = ["Malignant", "Benign"]
        return make_table(tab_dat,row_labels,col_labels,col_super=["Accuracy: "+str(acc_dat)+"%","Tumor ist"],row_super="Diagnostiziert als",)

    @render.plot()
    def clf_2D(): 
        
        data_points, decs_bound , x_dat, y_dat =  train_clf_bcw(features = [input.first2D_feature(),input.second2D_feature()])

        fig, ax = plt.subplots(1,1)
        ax.contourf(x_dat, y_dat, decs_bound, cmap = 'RdBu_r', alpha = 0.75 )
        scatter = ax.scatter(data_points[:,0],data_points[:,1], c = labels, cmap = cm_bright , edgecolors = 'k')
        handles = scatter.legend_elements()[0]
        ax.legend(handles, ['malignant','benign',],bbox_to_anchor=(1.0, 1.0), loc='upper left',  frameon = False)
        ax.set_xlabel(input.first2D_feature())
        ax.set_ylabel(input.second2D_feature())






