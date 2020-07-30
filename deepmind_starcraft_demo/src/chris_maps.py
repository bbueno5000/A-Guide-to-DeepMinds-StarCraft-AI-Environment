"""
Define the mini game map configs. These are maps made by Deepmind.
"""
import pysc2

class ChrisMaps(pysc2.maps.lib.Map):
    """
    DOCSTRING
    """
    directory = 'chris_maps'
    download = 'https://github.com/chris-chris/pysc2-examples#get-the-maps'
    players = 1
    score_index = 0
    game_steps_per_episode = 0
    step_mul = 8

chris_maps = ['DefeatZealots']

for name in chris_maps:
    globals()[name] = type(name, (ChrisMaps, ), dict(filename=name))
