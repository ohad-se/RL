
def traverse(goal_state, prev):
    '''
    extract a plan using the result of dijkstra's algorithm
    :param goal_state: the end state
    :param prev: result of dijkstra's algorithm
    :return: a list of (state, actions) such that the first element is (start_state, a_0), and the last is
    (goal_state, None)
    '''

    result = [(goal_state, None)]
    current_state = goal_state
    # print(len(prev.keys()))
    while prev[current_state.to_string()] is not None:
        prev_state = prev[current_state.to_string()]
        action = find_action(prev_state, current_state)
        result.insert(0, (prev_state, action))
        current_state = prev_state
    return result


def print_plan(plan):
    print('plan length {}'.format(len(plan)-1))
    for current_state, action in plan:
        print(current_state.to_string())
        if action is not None:
            print('apply action {}'.format(action))


def find_action(state, next_state):
    for action in state.get_actions():
        if state.apply_action(action) == next_state:
            return action
    return None
