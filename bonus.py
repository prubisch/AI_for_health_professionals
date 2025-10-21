from shiny import ui,render
from shiny.types import ImgData


bonus_page = ui.page_fluid(
    ui.HTML("<p>Unter der Trainingsmethode versteht man generell die Adaption der Parameter hinsichtlich der objektiven Funktion. Oft wird von optimalen Updates gesprochen. Alledings gibt es 3 verschiede Klassen von Trainingsalgorithmen:</p>"),
    ui.HTML("<ul><li>Überwacht/Supervised: Daten enthalten Klassenlabel</li><li>Reinforcement learning: Strategie-Lernen </li><li>Unüberwachtes/Unsupervised: Ohne Label</li></ul>"),
    ui.HTML("<p> <i>Note:</i>Da Reinforcement learning seine eigene Theorie und Updatestrategie besitzt, werden wir dies hier nicht weiter besprechen. Falls Sie doch daran interessiert sind empfehle ich Ihnen Resourcen zu Reinforcement-Learning/TD-Learning zu lesen.</p>"),
    ui.HTML("<p> Im Folgenden beschäftigen wir uns mit der <i>optimalen</i> Strategie die Parameter eines Systems zuadaptieren. Diese heißt <b>Gradient-descent</b>.</p>"),
    ui.HTML("<p> Gradient descent berechnet die <b>Richtung</b> in dem der <b>Fehler</b> reduziert wird. Das heißt wir suchen nach dem (globalen) Minima der Fehler-funktion. Die Abbildung unten visualisert dies: </p>"),
    ui.output_image("grad_des_ex", fill = True),
    ui.HTML("<p style='margin-top: -200px;'>Wie Sie sehen, hat sich der Fehler von Punkt I zu Punkt II durch das Parameterupdate um 1 verringert.</p><p>Mathematisch ist dies wie folgt zu beschreiben:</p>"),
    ui.HTML("<p> Das Parameterupdate $\Delta w_{ij}$ ist proportional zu der negativen Gradienten des Fehlers $E$ abhängig zu dem Parameter $w_{ij}$:</p> $$\Delta w_{ij} = - \eta \\frac{\partial E}{\partial w_{ij}}$$"),
    ui.HTML("<p>$\eta$ ist ein multiplikativer Faktor, der die Schrittgröße bestimmt. Er wird auch <i>Lernrate</i> genannt. Je größer $\eta$, desto größer sind die Parameter updates. Zu große Lernraten kann zu einer schlechten Performance führen, weil die gesuchten Minima übersprungen werden.</p>"),
    ui.HTML("<p>Unsere Fehlerfunktion $E$ (auch Loss genannt) wird hierbei durch unser Objective bestimmt. Im folgenden nutzen wir eine kontinuierliche Fehlerfunktion, den <b>MSE</b> zwischen den Output und dem Zielwert $t$, die wir schon in der Regression kennen gelernt haben. Damit ist unseren Fehlerfunktion definiert als:</p> $$E = \\frac{1}{n}\sum_i(t - o_i)^2$$"),
    ui.HTML("<p>$o_i$ ist der <b>Output</b> unseres Neuron $i$. Dieser ist abhängig von dem <b>Input</b> $net_i$ , der wiederum vom Gewicht/Parameter $w_{ij}$ und dem Output von Neuron $j$:</p> $$ net_i = \sum_j w_{ij}o_i$$ $$o_i = f(net_i)$$ <p> f(net_i) wird die <b> Aktivierungsfuntion</b> eines Neurons genannt. Es gibt verschieden Möglichkeiten für die Aktivierungsfunktion und diese kann auch neuronspezifisch sein in einen großen Netzwerk. Wichtig ist, dass die Aktivierungsfunktion differenzierbar ist.</p>"),
    ui.HTML("<p> Mit diesen Definition können wir uns jetzt die Differentialgleichung der Fehlerfunktion genauer anschauen:$$\\frac{\partial E}{\partial w_{ij}} = \\frac{\partial E}{\partial net_i}\\frac{\partial net_i}{\partial w_{ij}}$$</p>"),
    ui.HTML("<p> Aus der Definition des Input und Outputs und deren Zusamenhang, gilt:</p> $$ \\frac{\partial net_i}{\partial w_{ij}} = \\frac{\partial}{\partial w_{ij}}\sum_k o_k w_{ik} = o_j $$"),
    ui.HTML("<p> Wir definieren das Fehlersignal $\delta_i$ als:<p/>$$\delta_i = - \\frac{\partial E}{net_i}$$<p> Hierdurch können wir unser Parameterupdate umformulieren zu:</p>$$\Delta w_{ij} = \eta o_j \delta_i$$"),
    ui.HTML("<p>Das heißt, dass <i>das Parameterupdate ist das Produkt des Output des presynaptischen Neurons $j$ und dem Fehler des postsynaptischen Neurons $i$</i>.</p>"),
    ui.HTML("<p>Um das Parameterupdate errechnen zukönnen, muss auch $\delta_i$ hergeleitet werden: $$\delta_i = - \\frac{\partial E}{\partial net_i} = - \\frac{\partial E}{\partial o_i}\\frac{\partial o_i}{\partial net_i}$$"),
    ui.HTML("<p>Aus der Definition des Outputs und Inputs folgt:</p>$$\\frac{\partial o_i}{\partial net_i} = f'(net_i)$$"),
    ui.HTML("<p>Bei dem Differential des Fehlers nach dem Output des Neurons ist es wichtig im allgemeinen 2 Fälle zu unterscheiden:</p><ul><li>Output-Neuron (Neuron ist in der letzten Schicht/Layer):</li> $$\\frac{\partial E}{\partial o_i} = -(t - o_i)$$$$\delta_i = f'(net_i)(t-o_i)$$</ul><ul><li>Hidden-Neuron (Neuron aller anderen Schichten):</li></ul> $$\\frac{\partial E}{\partial o_i} = \sum_k \\frac{\partial E}{\partial net_k}\\frac{\partial net_k}{\partial o_i} = \sum_k - \delta_k \\frac{\partial}{\partial o_i}\sum_j o_j w_{kj} = \sum_k - \delta_k w_{ki}$$$$\delta_i = f'(net_i)\sum_k\delta_k w_{ki}$$"), 
    ui.HTML("<p> Das heißt, der Fehler der Output-Neurone muss zuerst berechnet werden, weil der Fehler aller nicht-Output-Neurone die Summe aller Fehler ihrer postsynaptische Neurone gewichtet mit dem Gewicht/Parameter der Verbindung ist. Während die Aktivität vorwärts durch das Netzwerk propagiert wird: Die Aktivierung der Neurone in den späteren Schichten ist abhängig von der Summe ihrer presynaptischen Neurone; werden die Fehler rückwärts durch das Netzwerk propagiert. Deswegen wird diese Art des Lernens mittels Gradient-descent auch <b>Backpropagation</b> genannt. </p>")
    
)

def bonus_images(input): 

    @render.image
    def grad_des_ex():
        from pathlib import Path
        dir = Path(__file__).resolve().parent
        img: ImgData = {"src": dir / "gradient_descent.png"}
        return img