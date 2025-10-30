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

# Your Task

Analyze the 8-directional observations to understand the spatial environment, then output a JSON defining the next navigation subtask.

**Core Requirements:**
1. **Spatial Understanding**: Determine your facing direction, identify obstacles/landmarks, estimate distances
2. **Path Planning**: Plan obstacle-aware navigation with concrete spatial movements (e.g., "turn right to face X, then move forward keeping Y on left")
3. **Subtask Design**: Choose an achievable intermediate destination (5-15 actions), ensure it's visible/inferable

**Output Quality**: Your JSON must demonstrate spatial reasoning through precise descriptions and realistic distance estimates.

# Output Format (Strict JSON)

**MUST output ONLY valid JSON - no markdown, no extra text**

**JSON Structure (7 Required Fields):**
{{
    "current_location": "Your position + facing direction + nearby landmark distances",
    "instruction_sequence": "Complete waypoint chain: A → B → C → D (final goal)",
    "subtask_destination": "The NEXT immediate waypoint",
    "subtask_instruction": "Concise action command to reach subtask_destination",
    "planning_hints": "**3 parts**: (1) Current facing direction, (2) Destination's relative position (e.g., 45° front-right), (3) **Obstacle navigation**: specific obstacles + how to avoid (e.g., 'pass table on left', 'go around couch')",
    "completion_criteria": "**3 constraints**: (1) Location: where agent should be, (2) Object: key objects visible, (3) Direction: spatial relationships. **NO IMAGE numbers, use general object terms**",
    "reasoning": "**3-part analysis**: (1) Spatial perception: facing direction (cite IMAGE #) + obstacle positions + distances, (2) Destination selection: why this waypoint + its relative position + distance estimate, (3) Path execution: concrete action sequence with obstacle avoidance (e.g., 'TURN_RIGHT once, MOVE_FORWARD 12 steps keeping table on left')"
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
    "planning_hints": "Currently facing TV wall (north). Kitchen doorway is at your front-right (45° angle). Coffee table is between you and doorway. Navigate by: turn right to align with doorway, then move forward keeping coffee table on your left side to avoid collision. Stop at doorway threshold before entering kitchen.",
    "completion_criteria": "Location: At kitchen doorway threshold, about to enter kitchen. Object: Kitchen doorway frame, coffee table, kitchen counters. Direction: Doorway and agent are close (within 1m), coffee table and agent are separated.",
    "reasoning": "(1) Spatial perception: Living room center facing TV wall (north) - TV in IMAGE 1 confirms facing direction, sofa in IMAGE 3 (right), coffee table between current position and kitchen doorway. Kitchen doorway visible in IMAGE 2 at 45° to my right. (2) Destination selection: Kitchen doorway is next in sequence (living room → doorway → kitchen → refrigerator), located at front-right position, ~3-4m distance estimate based on visual scale. (3) Path planning: From current north-facing orientation, need to turn right ~45° to face doorway direction. Coffee table obstacle requires navigating along its left edge. Concrete steps: TURN_RIGHT once to face doorway, then MOVE_FORWARD ~12 steps keeping coffee table on left, stop before crossing threshold."
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
    "planning_hints": "Currently facing south down the hallway corridor. Bedroom doorway is at your front-right (45° angle), ~4-5m away on the right wall. Navigate by: turn right to face the doorway, then move forward along right side of hallway keeping wall on your right. No obstacles between current position and doorway. Stop at doorway threshold.",
    "completion_criteria": "Location: At bedroom doorway threshold, about to enter bedroom. Object: Bedroom doorway frame, hallway corridor, bed. Direction: Doorway and agent are very close (within 1m), doorway dominates view, hallway corridor and agent are connected.",
    "reasoning": "(1) Spatial perception: Hallway entrance facing south - long corridor ahead in IMAGE 1 confirms facing direction, walls on both sides in IMAGE 3 (right) and IMAGE 7 (left). Bedroom doorway visible in IMAGE 2 at 45° to my right on east wall. (2) Destination selection: Bedroom doorway is next waypoint in sequence (hallway start → bedroom doorway → bedroom → bed), located at front-right position relative to current south-facing orientation, ~4-5m distance based on visual scale. (3) Path planning: From south-facing position, turn right ~45° to face doorway on east wall. Path is clear with no obstacles - hallway is open. Concrete steps: TURN_RIGHT once to align with doorway direction, then MOVE_FORWARD ~16 steps along right side of hallway keeping right wall as reference, stop at doorway threshold before entering bedroom."
}}

# Key Rules
- **Actions**: MOVE_FORWARD (0.25m/step), TURN_LEFT/RIGHT (45°/step), STOP
- **Subtask scope**: 5-15 actions, must be achievable from current position
- **IMAGE references**: Use in reasoning only (e.g., "IMAGE 1 shows..."), NOT in completion_criteria
- **Obstacle awareness**: Always specify facing direction + obstacle positions + concrete navigation (e.g., "turn right, move forward keeping table on left")
- **General object terms in completion_criteria**: "table" not "table on right", "door" not "door in front"
"""


# Verification prompt - checks if current subtask is completed
VERIFICATION_PROMPT = """Verify if the current subtask is completed based on visual observations.

# Navigation Context
**Goal**: {instruction}
**Current Subtask**: {subtask_description}
**Completion Criteria**: {completion_criteria}
**Planning Hints**: {planning_hints}

# Current Observations (8 Directions)
{direction_names}
IMAGE 1-8: Front(0°), Front-Right(45°), Right(90°), Back-Right(135°), Back(180°), Back-Left(225°), Left(270°), Front-Left(315°)

# Your Task
Compare observations with completion criteria → decide if completed → provide next step or guidance.

# Output Format (Strict JSON)

**MUST output ONLY valid JSON - no markdown, no extra text**

**JSON Structure (4 Required Fields):**
{{
    "is_completed": true/false,
    "completion_analysis": "Systematic check with IMAGE citations (e.g., 'IMAGE 1: doorway occupies 45% of view, distance <1.5m ✓. IMAGE 3: can see...')",
    "next_subtask": {{
        "subtask_instruction": "Next action command (if completed)",
        "planning_hints": "Facing direction + destination position + obstacle navigation",
        "completion_criteria": "3 constraints: Location + Object + Direction (general terms, no IMAGE #)"
    }},
    "continuation_advice": "If NOT completed: actionable guidance with IMAGE citations + action estimates (e.g., 'Continue FORWARD, IMAGE 1 shows door 3m away, need ~10 steps'). If completed: null"
}}

**Key Points**: 
- Cite specific IMAGE numbers in analysis
- If completed → design next logical subtask with spatial awareness
- If not completed → specify exact actions needed (TURN_LEFT/RIGHT, MOVE_FORWARD × N)
"""

# Task completion prompt - checks if entire navigation task is completed
TASK_COMPLETION_PROMPT = """Verify if the ENTIRE navigation task has been completed.

# Navigation Goal
{instruction}

# Current Observations (8 Directions)
{direction_names}
IMAGE 1-8: Front(0°), Front-Right(45°), Right(90°), Back-Right(135°), Back(180°), Back-Left(225°), Left(270°), Front-Left(315°)

# Your Task
Compare current observations with the goal's target location → determine completion confidence.

# Output Format (Strict JSON)

**MUST output ONLY valid JSON - no markdown, no extra text**

**JSON Structure (4 Required Fields):**
{{
    "task_completed": true/false,
    "confidence": 0.0-1.0,
    "analysis": "Systematic check for each goal element with IMAGE citations (e.g., 'Goal: \"chair next to fridge\". IMAGE 1: brown chair at 1m ✓. IMAGE 2: white fridge on right ✓. Position matches.')",
    "recommendation": "If NOT completed (confidence < 0.9): concrete next steps with IMAGE citations (e.g., 'Turn right to face kitchen, IMAGE 2 shows entrance'). If completed: null"
}}

**Confidence Scale**:
- 0.9-1.0: All goal elements visible, position perfect
- 0.7-0.9: Most elements match, minor adjustment needed
- 0.5-0.7: Some matches, significant differences
- < 0.5: Location doesn't match goal

**Key Rule**: Be conservative - only mark completed if confidence > 0.85
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
