import cv2
import sys
import time
sys.path.append("game/")
import wrapped_flappy_bird as game
# from BrainDQN_Nature import BrainDQN
import numpy as np
from RL_brain import DeepQNetwork


# preprocess raw image to 80*80 gray image
def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    #return np.reshape(observation,(6400,1))
    return observation.flatten()


def run_bird():
    step = 0
    for episode in range(100000):
        #print('episode:', episode)
        # initial observation
        # Step 3.1: obtain init state
        action0 = np.array([1, 0])  # do nothing
        observation0, reward0, terminal = env.frame_step(action0)
        #observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
        #ret, observation0 = cv2.threshold(observation0, 1, 255, cv2.THRESH_BINARY)
        #observation = np.reshape(observation0,(1,6400))
        observation = observation0.flatten()

        while True:
            # fresh env
            #env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)
            if action == 0: action = np.array([1, 0]);
            else: action = np.array([0, 1])

            # RL take action and get next observation and reward
            nextObservation, reward, done = env.frame_step(action)
            #nextObservation = preprocess(nextObservation)
            observation_ = nextObservation

            RL.store_transition(observation, action[1], reward, observation_)
            if (step > 50) and (step % 100 == 0):
                RL.learn()

            if (step > 50) and (step % 1000 == 0):
                print('Loss ', RL.get_cost())

            # swap observation
            observation = observation_
            # break while loop when end of this episode
            if done:
                break
            step += 1

            #if LastTime > 50: return(LastTime);

    # end of game
    print('game over')
    #env.destroy()


if __name__ == "__main__":
    # bird game
    env = game.GameState()  # define the game environment -> jump to game
    actions = 2
    features = 5
    RL = DeepQNetwork(actions, features,
                      learning_rate= 10**-2,
                      reward_decay=1.0,
                      e_greedy=0.6,
                      replace_target_iter= 200,
                      memory_size=50,
                      output_graph=True
                      )
    time.sleep(0.5)
    run_bird()
    #env.mainloop()
    RL.plot_cost()