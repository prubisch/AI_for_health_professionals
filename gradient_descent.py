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
sample_x = rng.rand(10)*2.8-1.4 #stretch it so it covers max and min and set the mean to 0
sample_errors = error_landscape(sample_x)[0]
eta = 1


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
        "Text explaining general waht we do",  
        ui.output_plot("plot_error_explanation"),
        ui.input_action_button("samples", "Nächste Schätzung"), 
        ui.input_action_button("reset", "Zurücksetzen"),
        "More text explaining",
        ui.output_plot("plot_error_landscape"),
        ui.input_action_button("left_step", "Kleiner"), 
        ui.input_action_button("right_step", "Größer"),
        ui.output_text("feedback"),
        
    )

def server_gradient_descent(input): 
    counter = reactive.Value(2)
    x_coord = reactive.Value(0)
    feedback_note = reactive.Value("")


    @reactive.effect
    @reactive.event(input.samples)
    def _(): 
        if counter.get() <10:
            counter.set(counter.get()+1)

    @reactive.effect
    @reactive.event(input.reset)
    def _():
        counter.set(2)


    @reactive.effect
    @reactive.event(input.left_step)
    def _(): 
        deriv = error_landscape(x_coord.get())[1]
        if deriv > 0: 
            x_coord.set(x_coord.get()-deriv * eta)
            feedback_note.set("Korrekt!")
        else: 
            feedback_note.set("Würde die Richtung wirklich den Fehler verringern?") #can i make this red and big? 


    @reactive.effect
    @reactive.event(input.right_step)
    def _(): 
        deriv = error_landscape(x_coord.get())[1]
        if deriv < 0: 
            x_coord.set(x_coord.get()-deriv * eta)
            feedback_note.set("Korrekt!")
        else: 
            feedback_note.set("Würde die Richtung wirklich den Fehler verringern?") #can i make this red and big? 

    
    @render.text
    def feedback():
        return feedback_note.get()


    @render.plot
    def plot_error_explanation(): 
        with plt.xkcd():
            fig, ax = plt.subplots(1,2)
            ax[0].scatter(sample_x[:counter.get()], error_landscape(sample_x[:counter.get()])[0])
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
                ax[1].text(0.25,0.9-i*0.1, str(np.round(sample_x[i],2)), ha = 'center', va = 'center')
                ax[1].text(0.75,0.9-i*0.1, str(np.round(sample_errors[i],2)), ha = 'center', va = 'center')
            ax[1].set_xticks([])
            ax[1].set_yticks([])



    @render.plot
    def plot_error_landscape():
        
        fig, ax = plt.subplots(1,1)
        ax.plot(gen_x, err_x)
        ax.set_xlabel('w')
        ax.set_ylabel('Fehler')
        ax.set_xlim(-1.4,1.4)
        ax.set_ylim(0,2)
        ax.scatter(x_coord.get(),error_landscape(x_coord.get())[0], color = 'red', marker = 'o', s = 20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)





