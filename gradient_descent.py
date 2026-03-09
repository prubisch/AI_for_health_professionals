import numpy as np
from shiny import ui,render, reactive
from shiny.types import ImgData
import matplotlib.pyplot as plt
import ast


def error_landscape(points):

    def error_fun(x): 
        return x**10 - x**7 - 5.07*x**6 + x**5  + 4.5*x**4 + 0.5*(x - 0.05)**3 - 0.1*x + 0.2
    def dx_fun(x): 
        return 10*x**9 - 7*x**6 - 6*5.07*x**5 + 5*x**4 + 18*x**3 + 1.5*(x-0.05)**2 - 0.1

    current_y = error_fun(points)
    current_dx = dx_fun(points)
    return current_y, current_dx



gen_x = np.linspace(-1.4,1.4, 100)
err_x = error_landscape(gen_x)[0]
rng = np.random.RandomState(seed = 42)

gradient_descent_page = ui.page_fluid(
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
        ui.HTML("<p>Neuronale Netzwerke und darauf basierende moderne KI-Systeme haben eine (sehr) hohe Anzahl an Parametern. Um die Performance des Systems/Netzwerkes zu verbessern lernt es. Im Endeffekt bedeuted das, dass die Parameter so geändert werden, dass der Fehler verringert wird. Lernverfahren haben daher die Parameteroptimierung als Ziel.</p>"), 
        ui.HTML("<p>Im folgenden ist es das Ziel, einen Paramter <b>$w$</b> zu optimieren.</p>"), 
        ui.HTML("<p>Die <i>naivste</i> Methode einen Parameter zu optimieren, ist zufällige Werte auszuwählen und diese zu evaluieren, also den Fehler zu berechnen.</p>"),
        ui.HTML("<p>Der folgende interaktive Graph und Tabelle berechnen 10 zufällige Werte.</p>"), 
        ui.HTML("<p> Welche Werte scheinen optimal zu sein? Ist dies ein gutes Vorgehen zur Parameteroptimierung? Wie verhält sich eine naive-randomisierte Suche, wenn man mehrere Parameter gleichzeitig optimieren will?</p>"), 
        ui.output_plot("plot_error_explanation"),
        ui.input_action_button("samples", "Nächste Schätzung"), 
        ui.input_action_button("reset", "Zurücksetzen"),
        ui.br(), ui.br(),
        ui.HTML("<p>Zufälliges oder auch strukturiertes Raten von Parameternwerten ist sehr langsam und unpräzise. Es gibt keine Garantie, dass man einen guten Wert gefunden hat geschweige denn den besten Wert. Der <b>minimale</b> oder <b>kleinst mögliche</b> Fehler ist unbekannt.</p>"),
        ui.HTML("<p>Es wäre vorteilhafter, wenn Techniken zur Parameteroptimierung iterativ den Parameterwert anpassen. Das heißt, wenn man errechnen könnte welche Veränderung (kleiner oder größerer Wert) den Fehler verkleinert. </p>"),
        ui.HTML("<p>Genau dies passiert im <i> Gradient Descent</i> Verfahren. Gradient Descent ist die Grundlage von vielen Parameteroptimierungen und lässt sich mathematisch rigoros definieren (siehe <i> Bonus</i> für mehr Informationen).</p>"),
        ui.HTML("<p> Hier soll die Intuition hinter Gradient Descent vermittelt werden. Dafür müssen zwei mathematische Konzepte definiert werden: Ableitungen und der Gradient.</p>"),
        ui.HTML("<p> Die Ableitung einer Funktion f(x) errechnet die <b>Steigung</b> an jeden Punkt beliebigen Punkt x der Funktion. Die Steigung berechnet, wie sich der Funktionswert f(x) verändert, wenn sich x vergrößert. Grafisch kann man dies als eine Tangente darstellen, die die Funktion in einem Punkt berührt. Die Intuition dahinter ist, dass eine Tangente, die lineare Approximation des Verhalten der Funktion an der Stelle $x$ ist. Also, wie die Funktion sich verhalten würde, wenn sie ab Punkt x als Linie weitergeführt werden würde. Grafisch kann dasdas wie folgt dargestellt werden:</p>"),
        ui.br(),
        ui.output_plot("gradient_demo"),ui.input_action_button("replay", "Demo wiederholen"),
        ui.HTML("<p>Die rote Linie stellt die Tangente und damit die Steigung von der in blau dargstellten Funktion $f(x)$ da.</p>"),
        ui.HTML("<p> Das heißt die Steigung zeigt an, in welche Richtung der Funktionswert kleiner wird: Ist die Steigung negativ, wird der Funktionswert kleiner, wenn x größer wird. Ist die Steigung positiv, wird der Funktionswert kleiner, wenn x kleiner wird. </p>"), 
        ui.HTML("<p>Der folgende Plot stellt weitere Beispielfuntkionen und deren Ableitungen dar."),
        ui.output_plot("plot_gradient_explanation"), ui.br(),
        ui.br(), ui.HTML("<p>Wenn wir Parameter optimieren, ist unsere Funktion f(x) die <b>Fehlerfunktion</b> oder <b>Fehlerlandschaft</b> und wird oft mit $E(x)$ bezeichnet.</p>"),
        ui.HTML("<p>Der Fehler und damit die Fehlerfunktion ist abhängig von <i>allen</i> Parametern. Damit ist die Fehlerfunktion multi-dimensional. Jeder Parameter steht für eine Dimension. Bei einer multi-dimensionalen Funktion nennt man die Steigung <b>Gradient</b>.</p>"),
        ui.br(), 
        ui.HTML("<p>Wieso interessiert uns der Gradient der Fehlerlandschaft?</p>"),
        ui.br(), ui.br(),
        ui.HTML("<p>Um die beste Performance zu erlangen, wollen wir die Parameterwerte finden bei denen der Fehler minimal ist. Das heißt, wir suchen nach dem <b>Minimum der Fehlerlandschaft</b>. Der Gradient hilft dabei ein Minimum einer Funktion zu finden:</p>"),
        ui.HTML("<p>Analytisch/Mathematisch ist jeder Punkt an dem der Gradient 0 ist ein <b>lokales</b> Minimum. Was ist der Unterschied zwischen einem lokalen und globale Minimum? Wo sind lokale und globale Minima in den Beispielfunktionen (Demo und Plots)? </p>"), 
        ui.br(),
        ui.HTML("<p>Bei Optimierungsproblemen kann der Gradient nicht exakt mit einer Funktion definiert werden. Daher kann man nicht analytisch die 0-Stellen der Ableitung/des Gradienten berechnen. Deswegen funktioniert Gradientdescent wie folgt: Wir folgen den (approximierten) Gradienten solange, bis sich das Vorzeichen des Gradienten ändert. Wenn der Parameterwert sich nur geringfügig ändert und der Gradient sehr sehr klein ist, kann man davon ausgehen, dass man ein Minimum erreicht hat.</p>"),
        ui.HTML("<p>Was <i>gerinfügig</i> und <si>sehr klein</i> konkret meint, definiert man im allgemeinen selber. Diese Kriterien, werden oft auch <i>Stopping-Kriterien</i> genannt und zählen zu den Hyperparametern.</p>"),
        ui.HTML("<p>Die folgende Grafik zeigt unsere Fehlerfunktion in Abhängigkeit zu dem Parameter $w$. Der große rote Punkt zeigt den aktuellen Parameterwert an. Die kleinen rötlichen Punkte visualisieren die letzten 10 Punkte der Optimierungshistorie.</p>"),
        ui.br(),
        ui.HTML("<p>Nutzen Sie die Pfeiltasten, um anzuzeigen in welcher Richtung sich der Fehler verbessert und den nächsten Schritt der Optimierung zu errechnen. Die Daten werden nur geupdated, wenn die Richtung der Parameteränderung korrekt angezeigt wurde.</p>"),
        ui.output_plot("plot_error_landscape"),
        ui.input_action_button("left_step", "$\Leftarrow$"), 
        ui.input_action_button("right_step", "$\\Rightarrow$"),
        ui.output_ui("feedback"),
        ui.input_action_button("restart", "Neuer Startwert"), 
        ui.HTML("<p>Was vermuten Sie passiert bei dem 2./3. Neustart, wenn die Optimierung in großen Schritten von links nach rechts sprint?</p>"), 
        ui.HTML("<p>Ein solches Verhalten, wenn die Optimierung persistent zwischen zwei Werten hin- und herspringt und sich über mehrere Interationen nicht verbessert, nenn man auch <i>nicht konvergent</i>. Auch wenn ein hin- und herspringen wie oben beschrieben, als zeichen gesehen werden kann, dass ein (lokales) Minimum identifiziert wurde, ist dies nicht immer der Fall. Um ein solches Verhalten zuvermeiden durch zu große Schritte, ist ein weiterer Hyperparameter die Lernrate. Mit der Lernrate skaliert man die 'Sprünge', die man während der Optimierung ausführt. Eine große Lernrate führt heufig zu nicht konvergenten Verhalten. Mit der Schaltfläche können Sie die Lernrate der Optimierung oben anpassen. Verringer sich die Häufigkeit vom Auftreten von nicht-konvergenten Verhalten? Kann eine Lernrate zu klein sein?</p>"),
        ui.input_select("eta", "Lernrate", choices = ["1","0.5","0.1","0.01","0.001"]),
        ui.HTML("<span style='color:black;font-size:16pt'>Gruppenaufgabe:</span>"), 
        ui.HTML("<p>Setzen Sie sich in 2-4er Teams zusammen. Jeder von Ihnen zeichnet eine Fehlerlandschaft. Tauschen Sie ihre Zeichnungen untereinander aus, sodass jeder nicht seine eigene erhält. Identifizieren Sie die lokalen und globalen Minima und zeichnen Sie 2 mögliche Optimierungsverhalten auf der Fehlerlandschaft ein. Die Startpunkte dürfen nicht IN den Minima liegen: Sie müssen mindestens 5 Schritte einzeichnen. </p>")

        
    )

def server_gradient_descent(input): 
    counter = reactive.Value(1)
    x_coord = reactive.Value(np.zeros(10))
    new_coord = reactive.Value(0)
    feedback_cor = reactive.Value(False)
    init_samples = rng.rand(10)*2.8-1.4
    sample_x = reactive.Value(init_samples)
    sample_errors = reactive.Value(error_landscape(init_samples)[0])
    timer_index = reactive.Value(0)
    x_range_demo = np.arange(-1.5,1.5, 0.05)
    


    @reactive.effect
    @reactive.event(input.samples)
    def _(): 
        if counter.get() <10:
            counter.set(counter.get()+1)

    @reactive.effect
    @reactive.event(input.reset)
    def _():
        counter.set(1)
        sample_x.set(rng.rand(10)*2.8-1.4)
        sample_errors.set(error_landscape(sample_x.get())[0])

    @reactive.effect
    @reactive.event(input.replay)
    def _(): 
        timer_index.set(0)

        


    @reactive.effect
    @reactive.event(input.left_step)
    def _(): 
        deriv = error_landscape(new_coord.get())[1]
        if deriv > 0: 
            new_coords = x_coord.get()
            new_coords[1:] = new_coords[0:-1]
            new_coords[0] = new_coords[1] - deriv * float(input.eta())
            if new_coords[0] < - 1.25: 
                new_coords[0] = -1.25
            x_coord.set(new_coords)
            feedback_cor.set(True)
            new_coord.set(new_coords[0])
        else:
            feedback_cor.set(False)
    
    @reactive.effect
    @reactive.event(input.right_step)
    def _(): 
        deriv = error_landscape(new_coord.get())[1]
        if deriv < 0: 
            new_coords = x_coord.get()
            new_coords[1:] = new_coords[0:-1]
            new_coords[0] = new_coords[1] - deriv * float(input.eta())
            if new_coords[0] > 1.35: 
                new_coords[0] = 1.35
            x_coord.set(new_coords)
            feedback_cor.set(True)
            new_coord.set(new_coords[0])
        else: 
            feedback_cor.set(False)
    @reactive.effect
    @reactive.event(input.restart)
    def _(): 
        new_coord.set(rng.rand(1)*2.6-1.25)
        x_coord.set(x_coord.get()*0+new_coord.get())
    
    @render.text
    def feedback():
        if feedback_cor.get(): 
            return ui.HTML("<span style = color:green;'> Korrekt. In dieser Richtung verbessert sich der Fehler!</span>")

        else: 
            return ui.HTML("<span style='color:red;font-size:20pt'>Würde die Richtung wirklich den Fehler verringern?</span>")


    @reactive.effect
    def _(): 
        reactive.invalidate_later(0.5)
        with reactive.isolate(): 
            i = timer_index.get()
            if i != -1 and i < len(x_range_demo)-1: 
                i = (i+1) % len(x_range_demo)
            else: 
                i = -1
            timer_index.set(i)



    @render.plot
    def plot_error_explanation(): 
        with plt.xkcd():
            fig, ax = plt.subplots(1,2)
            ax[0].scatter(sample_x.get()[:counter.get()], sample_errors.get()[:counter.get()])
            ax[0].set_xlabel('w')
            ax[0].set_ylabel('Fehler')
            ax[0].set_xlim(-1.4,1.4)
            ax[0].set_ylim(0,2)
            ax[0].spines['top'].set_visible(False)
            ax[0].spines['right'].set_visible(False)

            ax[1].spines['top'].set_visible(False)
            ax[1].spines['bottom'].set_visible(False)
            ax[1].spines['left'].set_visible(False)
            ax[1].spines['right'].set_visible(False)
            ax[1].plot([0,1],[1,1], color = 'k')
            ax[1].plot([0.5,0.5],[0,1.1], color = 'k')
            ax[1].text(0.25,1.05, 'w', ha = 'center', va = 'center')
            ax[1].text(0.75,1.05, 'Fehler', ha = 'center', va = 'center')
            for i in range(counter.get()): 
                ax[1].text(0.25,0.9-i*0.1, str(np.round(sample_x.get()[i],2)), ha = 'center', va = 'center')
                ax[1].text(0.75,0.9-i*0.1, str(np.round(sample_errors.get()[i],2)), ha = 'center', va = 'center')
            ax[1].set_xticks([])
            ax[1].set_yticks([])



    @render.plot
    def plot_error_landscape():

        fig, ax = plt.subplots(1,1)
        ax.plot(gen_x, err_x)
        ax.set_xlabel('w')
        ax.set_ylabel('E(x)')
        ax.set_xlim(-1.4,1.4)
        ax.set_ylim(0,2)
        points = x_coord.get()
        alpha_v = np.linspace(0.05,1,points.shape[0])[::-1]
        ax.scatter(new_coord.get(), error_landscape(new_coord.get())[0], color = 'red', marker = 'o', s = 30) 
        ax.scatter(points,error_landscape(points)[0], color = 'red', marker = 'o', s = 20, alpha = alpha_v)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


    @render.plot
    def plot_gradient_explanation(): 
        x_range = np.arange(-2,2, 0.01)

        fig, ax = plt.subplots(1,4, figsize = (4,12))
        l_tang = 0.5
        ax[0].plot(x_range, (x_range)**4 - 1.5 * x_range **2 + 0.5 * x_range**3, label = '$f(x) = x^4 + 0.5 x^3 - 1.5 x^2$')
        ax[0].plot([-1-l_tang,-1 + l_tang], [-1-l_tang*(0.5), -1 + l_tang*(0.5)], color = 'red', label = 'Ableitung/Gradient')
        ax[0].plot([1-l_tang,1 + l_tang], [0-l_tang*(2.5), 0 + l_tang*(2.5)], color = 'red')
        ax[0].plot([-l_tang,l_tang], [0-l_tang*(0), 0 + l_tang*(0)], color = 'red')
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].set_xlabel('x')
        ax[0].set_label('f(x)')
        ax[0].legend()


        ax[1].plot(x_range, x_range**2, label = '$f(x) = x^2$')
        ax[1].plot(x_range, x_range **2 + 3, label = '$f(x) = x^2 + 3$')
        ax[1].plot(x_range, x_range **2-0.5, label = '$f(x) = x^2 - 0.5$')
        #indicators for x = 1
        ax[1].plot(x_range, 2 * x_range, label = '$\dot{f(x)} = 2x$', color = 'red')
        ax[1].plot([1,1],[4,2], color = 'grey', linestyle = ':')
        ax[1].plot([1,1], [0.5,2], color = 'grey', linestyle = ':')
        #indicators for x = -1
        ax[1].plot([-1,-1],[4,-2], color = 'grey', linestyle = ':')
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].set_xlabel('x')
        ax[1].set_label('f(x)')
        ax[1].legend()


        ax[2].plot(x_range, (x_range+2)**2, label = '$f(x) = (x+2)^2$')
        ax[2].plot(x_range, (x_range+2) **2 + 3, label = '$f(x) = (x+2)^2 + 3$')
        ax[2].plot(x_range, (x_range+2) **2-0.5, label = '$f(x) = (x+2)^2 - 0.5$')
        #indicators for x = 1
        ax[2].plot(x_range, 2 * (x_range+2), label = '$\dot{f(x)} = 2(x+2)$', color = 'red')
        ax[2].plot([1,1],[12,6], color = 'grey', linestyle = ':')
        #indicators for x = -1
        ax[2].plot([-1,-1],[4,2], color = 'grey', linestyle = ':')
        ax[2].plot([-1,-1],[0.5,2], color = 'grey', linestyle = ':')
        ax[2].spines['top'].set_visible(False)
        ax[2].spines['right'].set_visible(False)
        ax[2].set_xlabel('x')
        ax[2].set_label('f(x)')
        ax[2].legend()


        ax[3].plot(x_range,  (x_range)**4 - 1.5 * x_range **2 + 0.5 * x_range**3, label = '$f(x) = x^4 + 0.5 x^3$')
        ax[3].plot(x_range, 4 * (x_range)**3 + 1.5 * x_range**2 - 3 * x_range, label = '$\dot{f(x)} = 4x^3 + 0.5 \cdot 3 x^2 - 2 \cdot 1.5 x$', color = 'red')
        #indicators for x = 1, y = 0, dx = 2.5
        ax[3].plot([1,1], [0, 2.5], color = 'grey', linestyle = ':')
        #indicators for x = -1, y = -1, dx = 0.5
        ax[3].plot([-1,-1], [-1, .5], color = 'grey', linestyle = ':')
        ax[3].spines['top'].set_visible(False)
        ax[3].spines['right'].set_visible(False)
        ax[3].set_xlabel('x')
        ax[3].set_label('f(x)')
        ax[3].legend()


    @render.plot
    def gradient_demo(): 
        x_range = np.arange(-2,2, 0.01)
        current_x = x_range_demo[timer_index.get()]

        fig, ax = plt.subplots(1,1)
        l_tang = 0.5
        ax.plot(x_range, (x_range)**4 - 1.5 * x_range **2 + 0.5 * x_range**3, label = '$f(x) = x^4 + 0.5 x^3 - 1.5 x^2$')
        dev = 4 * current_x ** 3 - 3 * current_x + 1.5 * current_x **2
        current_y = current_x ** 4 - 1.5 * current_x**2 + 0.5 * current_x **3
        ax.plot([current_x-l_tang,current_x + l_tang], [current_y-l_tang*(dev), current_y + l_tang*(dev)], color = 'red', label = 'Ableitung/Gradient')
        ax.set_title('x = '+str(round(current_x,2)) + '\n Steigung = '+str(round(dev,2)))
        ax.set_ylim(-2,14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel('f(x)')
        ax.set_xlabel('x')
        ax.legend()

