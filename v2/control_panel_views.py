from typing import Dict

import pprint
from new_guessing_game import run_simulations_in_parallel
from calculator import load_stimuli_and_calculator
from ipywidgets import widgets, HBox, VBox
from IPython.display import display, clear_output
from domain_objects import GameParams


def render_control_panel(simulation_result: Dict):
    style = {'description_width': 'initial'}

    stimuli_selection_widget = widgets.Select(options=['numeric', 'quotient'], value='quotient',
                                              description='Stimuli Type:', disabled=False)

    ans_selection_widget = widgets.Checkbox(value=True, description='with ANS')

    runs_selection_widget = widgets.IntSlider(
        value=2,
        min=1,
        max=10,
        step=1,
        description='Number of runs',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        style=style
    )

    population_size_selection_widget = widgets.IntSlider(
        value=10,
        min=2,
        max=20,
        step=2,
        description='Population size',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        style=style
    )

    steps_selection_widget = widgets.IntSlider(
        value=3000,
        min=100,
        max=5000,
        step=100,
        description='Number of steps',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        style=style
    )

    run_simulation_button = widgets.Button(
        description='Run Simulation',
        disabled=False,
        button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
        icon='play'
    )

    output = widgets.Output()

    def run_simulation_action(obj):
        run_simulation_button.disabled = True
        run_simulation_button.button_style = 'warning'
        run_simulation_button.icon = 'hourglass'
        run_simulation_button.description = 'Running ...'

        runs_selection_widget.disabled = True
        population_size_selection_widget.disabled = True
        ans_selection_widget.disabled = True
        steps_selection_widget.disabled = True
        stimuli_selection_widget.disabled = True

        runs = runs_selection_widget.value
        population_size = population_size_selection_widget.value
        stimulus = stimuli_selection_widget.value
        steps = steps_selection_widget.value
        with_ans = ans_selection_widget.value

        max_num = 100
        discriminative_threshold = 0.95
        discriminative_history_length = 50
        delta_inc = 0.2
        delta_dec = 0.2
        delta_inh = 0.2
        alpha = 0.01
        super_alpha = 0.001
        beta = 0.2
        guessing_game_2 = False
        seed = 100

        params = GameParams(population_size=population_size,
                            stimulus=stimulus,
                            max_num=max_num,
                            discriminative_threshold=discriminative_threshold,
                            discriminative_history_length=discriminative_history_length,
                            delta_inc=delta_inc,
                            delta_dec=delta_dec,
                            delta_inh=delta_inh,
                            alpha=alpha,
                            super_alpha=super_alpha,
                            beta=beta,
                            steps=steps,
                            runs=runs,
                            guessing_game_2=guessing_game_2,
                            seed=seed,
                            with_ans=with_ans)

        with output:
            clear_output()
            print('Running simulation with params:')
            printer = pprint.PrettyPrinter()
            printer.pprint(vars(params))

            stimuli, density, calculator = load_stimuli_and_calculator(params.stimulus, params.with_ans)

            populations = run_simulations_in_parallel(stimuli, calculator, params)

            run_simulation_button.button_style = 'success'
            run_simulation_button.disabled = False
            run_simulation_button.icon = 'play'
            run_simulation_button.description = 'Run Simulation'

            runs_selection_widget.disabled = False
            population_size_selection_widget.disabled = False
            ans_selection_widget.disabled = False
            steps_selection_widget.disabled = False
            stimuli_selection_widget.disabled = False

            simulation_result['stimuli'] = stimuli
            simulation_result['params'] = params
            simulation_result['populations'] = populations
            simulation_result['calculator'] = calculator
            simulation_result['density'] = density

    run_simulation_button.on_click(run_simulation_action)

    control_pane = HBox(
        [stimuli_selection_widget, ans_selection_widget,
         VBox([runs_selection_widget, population_size_selection_widget, steps_selection_widget])])

    display(VBox([run_simulation_button, widgets.HTML(value="<hr/>"), control_pane]), output)
