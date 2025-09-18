from parlai.core.params import ParlaiParser
from parlai.core.teachers import create_task_agent_from_taskname

# Setup options
parser = ParlaiParser(add_parlai_args=True, add_model_args=False)
opt = parser.parse_args([], print_args=False)
opt['task'] = 'dealnodeal'
opt['datatype'] = 'train'
opt['datapath'] = 'data/parlai'

# Create teacher
teacher = create_task_agent_from_taskname(opt)[0]

# Get a few examples
for i in range(3):
    example = teacher.act()
    print(f"\n=== Example {i+1} ===")
    print(f"Text: {example.get('text', '')[:200]}...")
    print(f"Labels: {example.get('labels', [])}")
    print(f"Episode done: {example.get('episode_done', False)}")
    print(f"Keys: {example.keys()}")
    teacher.observe({'episode_done': example.get('episode_done', False)})
    if teacher.epoch_done():
        break
