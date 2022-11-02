import numpy as np

Sdim_array = np.array([np.random.randint(2,18, size=100), np.random.randint(2,18, size=100), np.ones(100)], dtype=np.float32)
pos_array = np.array([np.random.randint(2,18, size=100), np.random.randint(2,18, size=100), np.ones(100)], dtype=np.float32)

def generate_obstacles(num_obstacles=2):
    obstacle_room = np.zeros((20 * 10, 20 * 10))

    obstacles = list()
    obstacle_room[:19, :] = 1
    obstacle_room[181:, :] = 1
    obstacle_room[:, :19] = 1
    obstacle_room[:, 181:] = 1

    for i in range(num_obstacles):
        obstacle_x = np.random.rand(1) * 0.8 + 0.1
        obstacle_y = np.random.rand(1) * 0.8 + 0.1
        size_x = np.random.rand(1) * 0.3 + 0.1
        size_y = np.random.rand(1) * 0.3 + 0.1

        position = [0, 0, 0, 0]  # x_low, x_high, y_low, y_high
        position[0] = int(obstacle_room.shape[0] * obstacle_x)
        position[1] = int(obstacle_room.shape[0] * (obstacle_x + size_x))
        if position[1] > obstacle_room.shape[0]:
            position[1] = obstacle_room.shape[0] - 1
        position[2] = int(obstacle_room.shape[1] * obstacle_y)
        position[3] = int(obstacle_room.shape[1] * (obstacle_y + size_y))
        if position[3] > obstacle_room.shape[1]:
            position[3] = obstacle_room.shape[1] - 1

        obstacle_room[position[0]: position[1], position[2]: position[3]] = 1
        obstacles.append(position)
    return obstacles, obstacle_room

obstacle_list = list()
obstacle_room_list = list()
for i in range(100) :
    while True :
        obstacle, obstacle_room=generate_obstacles()
        if obstacle_room[int(Sdim_array[0, i] * 10), int(Sdim_array[1, i] *10)] == 0 and obstacle_room[int(pos_array[0, i] * 10), int(pos_array[1, i] * 10)] == 0 :
            break
    obstacle_list.append(obstacle)
    obstacle_room_list.append(obstacle_room)
obstacle_list = np.array(obstacle_list)
obstacle_room_list = np.array(obstacle_room_list)

np.savez("test.npz", Sdim_array=Sdim_array, pos_array=pos_array, obstacle_list = obstacle_list, obstacle_room_list = obstacle_room_list)