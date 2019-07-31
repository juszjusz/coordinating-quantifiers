import sys

params = {"population_size": int(sys.argv[1]),
          #"weber_fraction_ratio": 0.3,  # used in noticeable_difference
          #"sigma": 0.1  # used in ReactiveUnit
          "discriminative_threshold": 0.95,
          "delta_inc": 0.1,
          "delta_dec": 0.1,
          "alpha": 0.1,  # forgetting rate
          "super_alpha": 0.01,  # complete forgetting of categories that have smaller weights
          "beta": 1.0,  # learning rate
          "steps": int(sys.argv[2]),
          "runs": 1}
