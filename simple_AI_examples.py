import numpy as np
from shiny import ui,render
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
#colorbling friendly plotting colours

 



def dataset_gen(N_train, N_test): 
    #we "model" the relationship between time studying to points on exam
    rng = np.random.default_rng(seed = 42)
    x_range = np.linspace(0, 120, N_train+N_test)
    gt = -0.006*( x_range-120)**2+90
    noise = rng.standard_normal(N_train+N_test)*10
    noisy_data = gt + noise
    shuffled_ind = rng.permutation(np.arange(0, N_train+N_test))
    train_data = [np.sort(x_range[shuffled_ind[:N_train]]), np.sort(noisy_data[shuffled_ind[:N_train]])]
    test_data = [np.sort(x_range[shuffled_ind[N_train:]]), np.sort(noisy_data[shuffled_ind[N_train:]])]
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


#i simply contruct the page in here and than return it
#"lineare Regression, k-means, Klassifikation"


simple_AI_page = ui.page_fluid(
    ui.navset_card_tab(
        ui.nav_panel("Regression", 
        "Regression ist die einfachste Form eines 'KI'-Algorithmus. Dieser versucht basierend auf den Trainingsdaten die grunlegende Beziehung zu identifizieren.",
        "Die Methode die dafür angewendet wird nennt sich 'means-square error (MSE) optimization'. Hierbei wird die Differenz zwischen den modellierten Daten und den echten",
        "echten Daten so klein wie möglich. Der MSE ist auch gleichzeitig die Metrik mit dem wir messen können wie gut unser Modell die Daten approximiert. ",
        "Der MSE ist definiert als: ",
        ui.p("$$MSE = \\frac{\\sum_n(y_i -(\hat{y}_i))^^2}{n}"),
        "Es berechnet die Differenz zwischen den Datenpunkten $$y_i$$ und den predizierten Punkten $$\\hat{y}_i$$ summiert diese auf und teilt sie durch die Anzahl",
        "an Datenpunkte. So berechnen wir die mittlere Abweichung zwischen den echten Daten und der Modellprädiktion. Es gilt: Je kleiner der MSE, desto besser die Approximation.",
        ui.output_plot("reg_explanation"),
        "Geben sind folgende (fiktive) Daten zu dem Effekt von Lernen auf die Verbesserung in der nächsten Klausur:", 
        "Wir teilen diese Daten in ein Trainings- und ein Testset ein. Das Trainingsset wird für die Modelloptimierung genutzt und das Testset wird zum errechnen eines Trainings-",
        "agnostischen Güte des Modells benutzt.", 
        ui.output_plot("vis_data"),
        "Um diese Daten mit einer Regression zu approximieren müssen wir auswählen, wie viele Parameter die Regression benutzt. Mit dem Slider lassen sich die Anzahl der Parameter ",
        "in der Regression einstellen.",
        ui.div(ui.div(ui.input_slider("fl1", "Anzahl Parameter", min = 1, max=20, value = 5), style = "text-align:center;"),style = "display:flex; justify-content:center;"),
        ui.output_plot("regression_plot"),
        "Frage: Würden Sie zustimmen, dass je kleiner der MSE ist, desto besser ist das Modell?",ui.br(),
        "Frage: Wie unterscheidet sich das Modellverhalten zwischen den Trainingsset und den Testset?",ui.br(),ui.br(),
        "Der Schieber unten zeigt Ihnen zusätzlich die Grundwahrheit der Daten an. Wie interpretieren Sie das Modellverhalten hinsichtlich dieser Information?",
        ui.input_switch("show_gt", "Grundwahrheit", False),), 

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
        ui.input_switch("show_gt_clustering", "Grundwahrheit", False),
        ),
        ui.nav_panel("Klassifikation", "simple classification example"),
    )
        
    )
 



#all render functions
def server_AI_examples(input): 
    import matplotlib.style as style
    style.use('seaborn-v0_8-colorblind')
    import matplotlib.pyplot as plt
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
        ax[1].scatter(train_dat[0], train_dat[1], color = palette[0], label = "Trainingsset")
        ax[1].scatter(test_dat[0], test_dat[1], color = palette[2], label = "Testset")
        ax[1].set_xlabel("Min. Lernen")
        ax[1].set_ylabel("% Verbesserung")


    @render.plot
    def regression_plot():

        dat_train, pred_train, dat_test, pred_test, real_data = simple_regression(input.fl1(), n_train, n_test)
        fig, ax = plt.subplots(1,2)
        ax[0].scatter(dat_train[0], dat_train[1], color = palette[0], label = 'Daten')
        ax[0].plot(dat_train[0], pred_train, color = palette[2], label = 'Model')
        if input.show_gt(): 
            ax[0].plot(real_data[0], real_data[1], color = 'k', label = 'Grundwahrheit')
        ax[0].set_title("Trainingsdaten")
        ax[1].scatter(dat_test[0], dat_test[1], color = palette[0], label = 'Daten')
        ax[1].scatter(dat_test[0], pred_test, color = palette[2], label = 'Model')
        if input.show_gt(): 
            ax[1].plot(real_data[0], real_data[1], color = 'k', label = 'Grundwahrheit')
        ax[1].set_title("Testdaten")
        ax[0].set_xlabel("Min. Lernen")
        ax[0].set_ylabel("% Verbesserung")
        ax[0].legend(bbox_to_anchor=(1.0, 1.0), loc='upper left',  frameon = False)
        ax[1].set_xlabel("Min. Lernen")
        ax[1].set_ylabel("% Verbesserung")
        ax[1].legend(bbox_to_anchor=(1.0, 1.0), loc='upper left',  frameon = False)

    @render.plot
    def reg_explanation(): 
        #load image I will have prepared outside
        plt.scatter(np.arange(10), np.arange(10))

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
            ax[1].set_title("Grundwahrheit")
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





