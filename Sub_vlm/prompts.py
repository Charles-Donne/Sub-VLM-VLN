"""
Prompt templates for LLM-based navigation planning
All prompts are in English for better LLM performance
"""

# Initial planning prompt - generates first subtask at task start
INITIAL_PLANNING_PROMPT = """You are the planning and thinking module of a Vision-Language Navigation agent. Your role is to analyze the spatial environment and design an easily achievable subtask for the low-level action execution module.

# Overall Navigation Task Instruction (Global Task)
{instruction}

# Current Visual Observations (8 Directional Views)
The agent is currently observing the environment from 8 directions (clockwise from front). Each image corresponds to:
{direction_names}

IMAGE 1: Front view (0°) - What the agent directly faces
IMAGE 2: Front-Right view (45°) - Diagonal front-right direction  
IMAGE 3: Right view (90°) - Direct right side
IMAGE 4: Back-Right view (135°) - Diagonal back-right direction
IMAGE 5: Back view (180°) - Behind the agent
IMAGE 6: Back-Left view (225°) - Diagonal back-left direction
IMAGE 7: Left view (270°) - Direct left side
IMAGE 8: Front-Left view (315°) - Diagonal front-left direction

# Task Description

Analyze the spatial environment from 8-directional observations and the global navigation instruction to output a structured JSON containing: current location, instruction sequence, subtask destination, subtask instruction, planning hints, completion criteria, and reasoning.

## Step 1: Spatial Understanding
Analyze ALL 8 images to determine:
1. **Current Location**: Identify where the agent is now based on room type, furniture, walls, and doors visible in the images
2. **Current Orientation**: Determine which direction the agent is facing based on what's in IMAGE 1 (front view)
3. **Navigation Sequence**: Parse the global instruction to extract ALL waypoints in order, then determine the NEXT waypoint to target

## Step 2: Subtask Destination Selection
Select the NEXT intermediate waypoint from the instruction sequence. Requirements:
1. Must be the immediate next location from the instruction sequence that is reachable
2. Must be visible or inferable from current 8-directional observations  
3. Must be achievable with 5-15 actions

# Output Requirements (Strict JSON Format)
{{
    "current_location": "Description of where the agent is with distances to key landmarks",
    "instruction_sequence": "Parsed waypoint sequence from global instruction (format: 'Location A → Location B → Location C (final)')",
    "subtask_destination": "The NEXT waypoint from the sequence",
    "subtask_instruction": "Action-oriented command: [ACTION] to [SUBTASK_DESTINATION]",
    "planning_hints": "High-level path: facing direction, relative position of destination, which side to pass obstacles",
    "completion_criteria": "THREE constraint types: (1) Location - where the agent should be positioned, (2) Object - what objects the agent should detect/see, (3) Direction - spatial relationships between objects and between objects and the agent",
    "reasoning": "Your systematic analysis in THREE parts: (1) Spatial perception - current location with image evidence and distances to landmarks, (2) Subtask destination - why this destination based on instruction sequence and visibility with distance estimate, (3) Path feasibility - why the path is executable considering obstacles and spatial layout"
}}

# Examples for Understanding

## Example 1: Living Room to Kitchen Task
**Global Instruction**: "Go to the kitchen and stop next to the refrigerator"
**Current Observation**: Agent in living room center, kitchen doorway visible in IMAGE 2 at 45° right

**Correct Output**:
{{
    "current_location": "Standing in living room center, facing north toward TV wall",
    "instruction_sequence": "Living room (current) → Kitchen doorway → Kitchen interior → Refrigerator location (final)",
    "subtask_destination": "Kitchen doorway entrance (threshold)",
    "subtask_instruction": "Turn right and move forward to the kitchen doorway entrance",
    "planning_hints": "Facing TV wall. Doorway is on your right front (45° angle). Pass by coffee table on your left. Stop before crossing threshold.",
    "completion_criteria": "Location: At kitchen doorway threshold, about to enter kitchen. Object: Kitchen doorway frame, coffee table, kitchen counters. Direction: Doorway and agent are close (within 1m), coffee table and agent are separated.",
    "reasoning": "(1) Current location: Living room center - TV in IMAGE 1 (front), sofa in IMAGE 3 (right side) confirm location. (2) Subtask destination: Kitchen doorway is next in instruction sequence (living room → kitchen doorway → kitchen → refrigerator), visible in IMAGE 2 at 45° right with ~3-4m distance. (3) Path planning: Turn right to face doorway, then move forward keeping coffee table on left provides clear, unobstructed path to destination."
}}

## Example 2: Hallway to Bedroom Task  
**Global Instruction**: "Walk down the hallway, enter the bedroom on the right, and stop beside the bed"
**Current Observation**: Agent in hallway start, bedroom door visible in IMAGE 2 at 45° right

**Correct Output**:
{{
    "current_location": "Standing at the beginning of hallway, facing forward (south) down the corridor",
    "instruction_sequence": "Hallway start (current) → Walk down hallway → Bedroom doorway on right → Bedroom interior → Bed location (final)",
    "subtask_destination": "Bedroom doorway entrance on the right",
    "subtask_instruction": "Turn right and move forward to the bedroom doorway entrance",
    "planning_hints": "Facing down hallway. Bedroom doorway is on your right front (45° angle). Keep to right side of hallway. Stop before entering bedroom.",
    "completion_criteria": "Location: At bedroom doorway threshold, about to enter bedroom. Object: Bedroom doorway frame, hallway corridor, bed. Direction: Doorway and agent are very close (within 1m), doorway dominates view, hallway corridor and agent are connected.",
    "reasoning": "(1) Current location: Hallway start - long corridor in IMAGE 1 (front), walls on both sides in IMAGE 3/7 confirm position. (2) Subtask destination: Bedroom doorway is next in instruction sequence (hallway start → bedroom doorway → bedroom → bed), visible in IMAGE 2 at 45° right with ~4-5m distance. (3) Path planning: Turn right to face doorway, then move forward along right side of hallway provides direct access without obstacles, matching instruction's 'bedroom on the right'."
}}

# Key Constraints
- Agent capabilities: MOVE_FORWARD (0.25m/step), TURN_LEFT/RIGHT (45°/step), STOP
- Subtask must be completable in 5-15 actions
- Must reference specific IMAGE numbers in reasoning for current observation analysis
- Completion criteria must include all three constraint types WITHOUT IMAGE references (verifiable in any future observation)
- **CRITICAL**: In completion_criteria, describe landmarks and objects in GENERAL terms without precise directions or order:
  * "second step from top" → "step" (ignore order/position)
  * "room in front" → "room" (ignore direction)
  * "bedroom door on the right" → "bedroom door" (ignore side)
  * Focus on WHAT objects exist, not WHERE they are relative to agent
"""


# Verification prompt - checks if current subtask is completed
VERIFICATION_PROMPT = """You are the planning and thinking module of a Vision-Language Navigation agent. You need to verify if the current subtask is completed and guide the next step.

# Overall Navigation Goal
{instruction}

# Current Subtask Being Verified
- Description: {subtask_description}
- Completion Criteria: {completion_criteria}
- Planning Hints: {planning_hints}

# Current Visual Observations (8 Directional Views)
The agent is now observing from 8 directions (clockwise from front). Each image corresponds to:
{direction_names}

IMAGE 1: Front view (0°) - What the agent directly faces
IMAGE 2: Front-Right view (45°) - Diagonal front-right direction  
IMAGE 3: Right view (90°) - Direct right side
IMAGE 4: Back-Right view (135°) - Diagonal back-right direction
IMAGE 5: Back view (180°) - Behind the agent
IMAGE 6: Back-Left view (225°) - Diagonal back-left direction
IMAGE 7: Left view (270°) - Direct left side
IMAGE 8: Front-Left view (315°) - Diagonal front-left direction

# Your Mission
1. Compare the 8 current observations with the completion criteria
2. Determine if the subtask is completed
3. If COMPLETED → Design the next subtask (similar format as before)
4. If NOT COMPLETED → Provide specific guidance for execution module

# Output Requirements (Strict JSON Format)
{{
    "is_completed": true/false,
    "completion_analysis": "Detailed analysis referencing specific images (e.g., 'In IMAGE 1, the door now occupies 45% of view and distance appears <1.5m, meeting criteria. In IMAGE 3, I can see...')",
    "next_subtask": {{
        "subtask_description": "Next subtask if current completed",
        "planning_hints": "Execution guidance (direction, landmarks, actions needed)",
        "completion_criteria": "Observable verification criteria"
    }},
    "continuation_advice": "If NOT completed: specific advice with image references (e.g., 'Continue FORWARD. IMAGE 1 shows door is still 3m away, need 8-10 more steps. Watch for chair in IMAGE 2'). If completed: null"
}}

# Key Points
- Reference specific IMAGE numbers when analyzing observations
- Compare visual evidence with completion criteria systematically
- If generating next subtask, ensure logical progression toward overall goal
- Advice should be actionable: which actions, how many times, what to watch for
"""

# Task completion prompt - checks if entire navigation task is completed
TASK_COMPLETION_PROMPT = """You are the planning and thinking module of a Vision-Language Navigation agent. You need to verify if the ENTIRE navigation task has been successfully completed.

# Overall Navigation Goal (Target to Reach)
{instruction}

# Current Visual Observations (8 Directional Views)
The agent is now at a position, observing from 8 directions (clockwise from front):
{direction_names}

IMAGE 1: Front view (0°) - What the agent directly faces
IMAGE 2: Front-Right view (45°) - Diagonal front-right direction  
IMAGE 3: Right view (90°) - Direct right side
IMAGE 4: Back-Right view (135°) - Diagonal back-right direction
IMAGE 5: Back view (180°) - Behind the agent
IMAGE 6: Back-Left view (225°) - Diagonal back-left direction
IMAGE 7: Left view (270°) - Direct left side
IMAGE 8: Front-Left view (315°) - Diagonal front-left direction

# Your Mission
Carefully compare the TARGET LOCATION described in the goal with ALL 8 current observations to determine:
- Does the current location match the target description?
- Are all key landmarks/objects mentioned in the goal visible?
- Is the agent positioned as described?

# Output Requirements (Strict JSON Format)
{{
    "task_completed": true/false,
    "confidence": 0.0-1.0,
    "analysis": "Systematic analysis with image references. For each key element in the goal, state which IMAGE(s) show evidence. Example: 'Goal mentions \"chair next to fridge\". IMAGE 1 shows a brown chair at 1m distance. IMAGE 2 shows white fridge immediately to the right. Position matches goal description.'",
    "recommendation": "If NOT completed (confidence < 0.9): Specific next steps (e.g., 'Turn right to face the kitchen mentioned in goal' or 'Move forward 5 steps, IMAGE 1 shows target room entrance ahead'). If completed: null"
}}

# Evaluation Guidelines
- Confidence = 0.9-1.0: All goal elements clearly visible in images, position matches perfectly
- Confidence = 0.7-0.9: Most elements match, minor positioning adjustment may be needed  
- Confidence = 0.5-0.7: Some elements match, but significant differences remain
- Confidence < 0.5: Current location does not match goal description
- Always reference specific IMAGE numbers as evidence
- Be conservative: only mark completed if highly confident (>0.85)
"""


def get_initial_planning_prompt(instruction: str, direction_names: list) -> str:
    """
    Get initial planning prompt
    
    Args:
        instruction: Complete navigation instruction
        direction_names: List of direction names
        
    Returns:
        Formatted prompt string
    """
    return INITIAL_PLANNING_PROMPT.format(
        instruction=instruction,
        direction_names=', '.join(direction_names)
    )


def get_verification_prompt(instruction: str, 
                           subtask_description: str,
                           completion_criteria: str,
                           planning_hints: str,
                           direction_names: list) -> str:
    """
    Get verification prompt
    
    Args:
        instruction: Complete navigation instruction
        subtask_description: Current subtask description
        completion_criteria: Completion criteria
        planning_hints: Planning hints
        direction_names: List of direction names
        
    Returns:
        Formatted prompt string
    """
    return VERIFICATION_PROMPT.format(
        instruction=instruction,
        subtask_description=subtask_description,
        completion_criteria=completion_criteria,
        planning_hints=planning_hints,
        direction_names=', '.join(direction_names)
    )


def get_task_completion_prompt(instruction: str, direction_names: list) -> str:
    """
    Get task completion check prompt
    
    Args:
        instruction: Complete navigation instruction
        direction_names: List of direction names
        
    Returns:
        Formatted prompt string
    """
    return TASK_COMPLETION_PROMPT.format(
        instruction=instruction,
        direction_names=', '.join(direction_names)
    )
