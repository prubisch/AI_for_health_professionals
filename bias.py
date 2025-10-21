from shiny import ui, render
from shiny.types import ImgData
import numpy as np
import matplotlib.pyplot as plt

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

bias_page = ui.page_fluid(
    ui.navset_card_tab(
        ui.nav_panel("Was ist Bias?", 
        ui.HTML("<p>Wie die Beispiele der Regressions, des Clusterings und der Klassifikation gezeigt haben ist das Ziel von KI Systemen <b> statische Zusammenhänge</b> im Datensatz zu identifizieren. Daher ist die Güte und Neutralität des Datensatzes zentral.</p>"),
        ui.HTML("<p>Wenn die Daten selber Korrelationen beinhalten, die z.B. durch die Messungstechnik entstehen, also 3. Quellen können KI-Systeme diese identifizieren bzw. ausnutzen. Das heißt es können Zusammenhänge erlernt werden, die die Performance des Systems erhöhen, es aber weniger <i>robust</i> machen. Diese unbeabsichtigen oder versteckten Korrelationen nennt man <b>Bias</b>. </p>"),
        ui.HTML("<p>Bias kann durch verschiedene Quellen entstehen. Wie oben genannt, kann es ein systemischer Einfluss im Datensatz selber sein, z.B. durch <b>unsaubere</b> Datenerhebung. Bei der Klassifizierung kann es imbesonderen zum Bias kommen, wenn Klassen <b>unterrepräsentiert</b> sind.</p>"),
        ui.HTML("<p>In kurz: <i> Bias ist eine Verzerrung der zugrundlegenden statistischen Zusammenhänge im Datensatz durch externe Einflüsse</i>.</p> "), ui.br(), ui.br(),
        ui.HTML("<p>Bevor Sie sich durch die beiden folgende Beispiele durcharbeiten: Welche Biasquellen können in medizinischen Datensätzen auftreten? Welche Biases sind insbesondere im Gesundheitsmanagement zu beachten?</p>"), 
        ),
        ui.nav_panel("Unterrepräsentation einer Klasse",
        ui.HTML("<p> Um die Effekte von Bias in den Trainingsdaten zu explorieren, nutzen wir ein Standardproblem im Bereich des Maschinellen Lernens: <b> Die Klassifizierung von CIFAR-10 Bildern </b>. CIFAR-10 ist ein Datenset mit natürlichen Bildern und ihren dazugehörigen Klassen. Es enthält 10 Klassen: <i> Airplane, Car, Bird, Cat, Deer, Dog, Frog, Horse, Ship und Truck </i>. Der Standard-Trainings- und Testset besteht aus 50000 und 10000 Bilder. Jedes Bild ist 32x32 Pixel groß. Hier können Sie 25 Beispielbilder sehen inklusive ihrer Klassen.</p>"), 
        ui.br(),
        ui.output_image("CIFAR10",fill=True),
        ui.br(),
        ui.br(),
        ui.br(),
        ui.br(),
        ui.HTML("<p><i>Wie schätzen Sie diese Menge an Trainingsdaten ein? Ist das viel oder wenig für die 10 Klassen von natürlichen Bildern? </i></p>"),
        ui.br(), ui.br(),
        ui.HTML("<p> Wie Sie sehen können sind die Bilder sehr divers und es sind viele verschiedene Aufnahmewinkel und Positionen innerhalb der ersten 25 Bilder in der Datenmenge zu sehen. Das macht die Klassifizierung ein komplexes Problem. </p>"), 
        ui.HTML("<p> Im Folgenden benutzen wir wieder ein <b> Neuronale Netzwerk </b> um dieses Problem zu lösen. Allerdings müssen wir die Größe des Netzwerkes unserem Problem anpassen: Das Netzwerk, was wir im trainieren hat 7 Layers und insgesamt 62006 trainierbare Parameter. Die Architektur ist die eines <i> Convolutional Neural Network</i> (CNN), und basiert auf dem LeNet (LeCun et al., 1998).</p>"),
        ui.HTML("<p><i>Hinweis:</i> Je größer ein Netzwerk, desto mehr Zeit und Energie wird benötigt, um es zu trainieren. Wir benutzen eine kleinere und inzwischen veraltete Architektur, da die Probleme durch inadäquate Trainingsdaten unabhängig von der Architektur sind.</p>"),
        ui.HTML("<p>Wir trainieren zuerst unser CNN mit den Standard-Datesatz, um einen Vergleichswert neben dem Zufallslevel von 10% zu haben. Die nachfolgende Grafik zeigt Ihnen die generelle Accuracy (Titel) und eine Confusion-Matrix. Die Confusion-Matrix zeigt die Aufteilung pro Klasse der Prediktionen. Daher können Sie klassenweise einsehen, mit welcher anderen Klasse die Testbilder am häufigsten verwechselt (also confused) werden.</p>"),
        ui.output_plot("standard_confusion"),
        ui.HTML("<p> Wie schätzen Sie diese Performance ein (z.B. im Vergleich zum Zufallsniveau)? Woran könnte es liegen, dass Hunde zu 26 % als Katzen misklassifiziert werden? </p>"), 
        ui.HTML("Die Trainingsdaten enthalten die gleiche Anzahl an Bildern von jeder Klasse. Wie beeinflusst es die Performance, wenn wir die Anzahl der Katzen bilder auf 20% der Anzahl jeder anderen Klassenbeispiele reduzieren?</p>"), 
        ui.br(),
        ui.HTML("<p> Mit dem Schieber trainieren Sie ein Netzwerk mit der selben Architektur wie unser Standard-CNN. Allerdings ist die Anzahl der Katenbilder auf 20% reduziert in unseren Traininsdaten. Vergleichen Sie die Performance und Confusionmatrix von diesem Red-CNN zum Standard-CNN."),
        ui.input_switch("show_reduced", "Trainiere & Test Red-CNN", False),
        ui.output_plot("reduced_confusion"), 
        ui.HTML("<p>Denken Sie an den Leitspruch '<i>Wenn Sie Hufgeräusche hören, denken Sie an Pferde nicht Zebras.</i>' Was bedeuten Zebras für die Trainings- und Testdatenverteilung? </p>"),
        ui.HTML("<p>Denken Sie an die Misklassifizierung im Standard-CNN zurück. Denken Sie diese Fälle können auch bei der Klassifizierung von psychologischen Krankheiten auftreten? Warum? </p>")
        ),

        ui.nav_panel("Unsaubere Daten",
        ui.HTML("<p> Das Standard-CIFAR-10 Datenset ist frei von 'Messfehlern' oder Verunreinigungen, daher fügen wir künstlich eine systemische Verunreinigung in unsere Trainingsdatensatz ein. Hier können Sie 25 Beispielbilder inklusive ihrer Klassen sehen. Können Sie die Verunreinigung identifizieren?</p>"), 
        ui.br(),
        ui.output_image("Spot10",fill=True),
        ui.br(),
        ui.br(),
        ui.br(),
        ui.br(),
        ui.HTML("<p> Nutzen Sie den Schieber um sich 5 weitere Beispielbilder einer bestimmten Klasse anzeigen zu lassen. Finden Sie eine nichtkausale Korrelation zwischen den Bildern?</p>"), 
        ui.input_switch("show_cats", "Beispiele einer Klasse", False),
        ui.output_image("corrupted",fill=True),
        ui.br(), ui.br(),
        ui.HTML("<p><i>Wie wird sich diese Manipulation auf die Performance und die Confusion-Matrix auswirken? Was erwarten Sie? </i></p>"),
        ui.HTML("<p>Da wir jetzt 2 Test-sets zur Verfügung haben, das Standard-Testset und das verunreinigte Testset, sollten sie sich die Performance auch auf beiden Sets anschauen. Mit dem Auswahlmenü können Sie zwieschen den Ergebnissen wechseln.</p>"),
        ui.input_select("test_set_choice", "Testset:", ["Verunreinigt","Standard"] ),
        ui.output_plot("confusion_adversarial"), 
        ui.br(),
        ui.HTML("<p> Im folgenden wollen wir unser neues neu-trainiertes Netzwerk weiter testen und geben ihm folgende Bilder: </p>"), 
        ui.output_image("corrupted_test",fill=True),
        ui.HTML("<p>Schreiben Sie sich ihre Vermutungen zu den predizierten Klassen auf. Benutzen Sie den Schieber um sich die Klassifizierung von unseren 'Adversarial'-CNN anzeigen zu lassen.</p>"),
        ui.input_switch("show_pred", "Zeige Prediktion", False),
        ui.HTML("<p><i>Info:</i> Adversarial attacks, randomisierte (oder auch systemische) Manipulationen von den Testdaten werden genutzt, um die Robustheit von KI-Systeme wie CNNs oder Transformer zu testen. Warum ist es wichtig, dass Systeme gegen Adversarialattacks robust sind? </p>"),
        )
        
    ))


def plot_confusion_matrixes(path_mat):

    mat_dat = np.load(path_mat)

    fig, ax = plt.subplots(1,1)
    ax.imshow(mat_dat['confusion_matrix'], aspect = 'auto', cmap = 'viridis') 
    for (i, j), value in np.ndenumerate(mat_dat['confusion_matrix'].T):
        ax.text(i, j, "%.1f"%value, va='center', ha='center')
    ax.set_xticks(np.arange(len(classes)), classes)
    ax.set_yticks(np.arange(len(classes)), classes)
    ax.set_ylabel('Class')
    ax.set_xlabel('Predicted as')
    ax.set_title('Accuracy:'+str(mat_dat['performance'])+'%')
    
    return fig


def server_bias(input): 
    import matplotlib.style as style
    style.use('seaborn-colorblind')
    
    @render.image
    def CIFAR10():
        from pathlib import Path
        dir = Path(__file__).resolve().parent
        img: ImgData = {"src": dir / "example_images/standard_images.png"}
        return img

    @render.image
    def Spot10():
        from pathlib import Path
        dir = Path(__file__).resolve().parent
        img: ImgData = {"src": dir / "example_images/adversarial_images.png"}
        return img

    @render.image
    def corrupted():
        if input.show_cats(): 
            from pathlib import Path
            dir = Path(__file__).resolve().parent
            img: ImgData = {"src": dir / "example_images/cats_corrupted_no_pred_24.png"}
            return img

    @render.image
    def corrupted_test():
        if input.show_pred(): 
            from pathlib import Path
            dir = Path(__file__).resolve().parent
            img: ImgData = {"src": dir / "example_images/unconditional_corrupted_24.png"}
            return img
        else: 
            from pathlib import Path
            dir = Path(__file__).resolve().parent
            img: ImgData = {"src": dir / "example_images/unconditional_corrupted_no_pred_24.png"}
            return img


    @render.plot
    def standard_confusion():

        plot_confusion_matrixes('trained_networks/standard_trained_standard_images_confusion_matrix.npz')

    @render.plot
    def reduced_confusion():
        if input.show_reduced(): 
            plot_confusion_matrixes('trained_networks/unbalanced_standard_images_confusion_matrix.npz')


    @render.plot
    def confusion_adversarial():
        if input.test_set_choice()== "Standard": 
            plot_confusion_matrixes('trained_networks/adversarial_trained_standard_images_confusion_matrix.npz')
        elif input.test_set_choice()== "Verunreinigt":
            plot_confusion_matrixes('trained_networks/adversarial_trained_adversarial_images_confusion_matrix.npz')