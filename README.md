# FlappyDML

# Usage

## Training a Model

Train a model using the learn command:
```python
python script.py learn --name my_model --alg PPO --eps 200 --timestamps 100000 --verbose 1

    #--name: (Optional) Specify the name of the model. Default is "default".
    #--alg: (Optional) Specify the algorithm to use. Default is "PPO".
    #--eps: (Optional) Number of episodes for training. Default is 200.
    #--timestamps: (Optional) Timestamps per episode. Default is 100000.
    #--verbose: (Optional) Verbose level. Default is 1.
```

## Displaying Results

Display results using the display command:

```python
python script.py display --name my_model --timestamp latest --eps 10

    #--name: (Optional) Specify the name of the model. Default is "default".
    #--timestamp: (Optional) Specify the timestamp parameter. Default is "latest".
    #--eps: (Optional) Number of episodes to display. Default is 10.
```

## Playing a Game

Play a game using the play command:

```python
python script.py play --debug

   # --debug: (Optional) Enable debug mode.
```

## Global Debug Mode

To enable global debug mode for any command, use the --debug flag:

```python
python script.py learn --debug
python script.py display --debug
python script.py play --debug
```