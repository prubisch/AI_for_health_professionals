import seaborn as sns

# Import data from shared.py
# for modularisation we use the shiny core syntax
#from shared import df
from shiny import App, ui, render, reactive
from simple_AI_examples import simple_AI_page, server_AI_examples


app_ui = ui.page_navbar(


    ui.nav_panel("Grundbausteine von KI", "Objektiv, Trainingsdaten, Architektur, Training Algorithmus"
    ),
    ui.nav_panel("Einfach(st)e KI Beispiele", simple_AI_page),
    ui.nav_panel("Bias in den Trainingsdaten", "Unzureichende Datenmengen, ausgelassene Klassen, Korrelationen "), 
    ui.nav_panel("Katastrophales Vergessen", "Kontinuierliches Lernen und One-shot Lernen"),
    title = "KI für Gesundheitsberufe", 
    position = 'static_top'
)


def server(input, output, session): 
    #all reactive and render functions go here
    server_AI_examples(input)
    


app = App(app_ui, server)