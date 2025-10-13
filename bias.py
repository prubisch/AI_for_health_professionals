#classification
from shiny import ui



bias_page = ui.page_fluid(
    ui.navset_card_tab(
        ui.nav_panel("Was ist Bias?", "KI Systeme und Algorithmen erlernen statistische Zusammenhänge. Das heißt, die Daten die für das Training benutzt werden un daher ihre Statistiken sind ein zentrales Stück jedens KI-Systems.",
        "Daher ist es wichtig, dass die Daten gewisse Kriterien erfüllen und brauchbar sind. Ansonsten können sich Biases einschleichen, die nur bedingt durch zusätzliche Optimierungskriterien ausgleichen lassen.",
        "Als Bias bezeichnen wir die Verzerrung der grundlegenden statistischen Zusammenhänge in den Trainingsdaten. ", ui.br(), ui.br(),
        "Aufgabe: Welche Biases können in medizischen Datensätzen vorhanden sein?", 
        "Nennen Sie 2 konkrete Beispiele für mögliche Biases bei KI-basierten Anwendungen für Gesundheitsberufe, z.B. in der Diagnostik. "
        ), 

        ui.nav_panel("Unterrepräsentation einer Klasse"
        ),

        ui.nav_panel("Unsaubere Daten",)
        
    ))



def server_bias(input): 
    import matplotlib.style as style
    style.use('seaborn-v0_8-colorblind')
        




