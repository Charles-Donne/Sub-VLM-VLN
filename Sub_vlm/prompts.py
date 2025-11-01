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

IMAGE 1: Front view (0°) - What the agent directly faces (PRIMARY view for planning)
IMAGE 2: Front-Right view (45°) - Diagonal front-right direction  
IMAGE 3: Right view (90°) - Direct right side
IMAGE 4: Back-Right view (135°) - Diagonal back-right direction
IMAGE 5: Back view (180°) - Behind the agent
IMAGE 6: Back-Left view (225°) - Diagonal back-left direction
IMAGE 7: Left view (270°) - Direct left side
IMAGE 8: Front-Left view (315°) - Diagonal front-left direction

# Task Requirements

Analyze the 8-directional observations to understand the spatial environment, then output a JSON defining the next navigation subtask.

**Core Requirements:**
1. **Spatial Analysis**: Identify key landmarks, obstacles, and spatial relationships
2. **Start planning from IMAGE 1 (current front view)** - this is your first-person perspective
3. **Action-Oriented Planning**: Use concrete actions - turn toward [landmark], move to [object], navigate around [obstacle], rotate to face [target]
4. **Landmark-Based Reference**: Anchor all descriptions to visible objects/landmarks, avoid abstract directions
5. **Output Style**:
   - current_location: Concise position relative to 1-2 key landmarks
   - subtask_destination: Brief target description with landmark reference
   - subtask_instruction: Action sequence starting from IMAGE 1 (3-5 actions max)
   - planning_hints: Detailed navigation strategy from first-person perspective
   - completion_criteria: 3 object-based constraints (NO left/right/front/back - use "near [object]", "between [A] and [B]", "facing [landmark]")

**Output Quality**: Provide concise guidance that captures key spatial relationships without over-specification.

# Output Format (Strict JSON)

**MUST output ONLY valid JSON - no markdown, no extra text**

**JSON Structure (7 Required Fields):**

{{
    "current_location": "Position summary with 1-2 nearby landmarks",
    "instruction_sequence": "Waypoint chain: A → B → C → D (final goal)",
    "subtask_destination": "The NEXT immediate waypoint",
    "subtask_instruction": "Action sequence: turn toward X, move to Y, navigate around Z",
    "planning_hints": "Detailed strategy: (1) Current facing + key obstacles, (2) Target relative position via landmarks, (3) Navigation path with actions (turn toward, move along, rotate to face, navigate around)",
    "completion_criteria": "3 object-based checks: (1) Proximity to specific object, (2) Observable landmarks, (3) Spatial relationship (between X and Y, near Z, facing toward W)",
    "reasoning": "4-part logic: (1) Spatial perception from images, (2) Destination choice + landmark reference, (3) Action strategy, (4) Consistency check"
}}

# Examples for Understanding

## Example 1: Living Room to Kitchen Task
**Global Instruction**: "Go to the kitchen and stop next to the refrigerator"
**Current Observation**: Agent in living room center, kitchen doorway visible in IMAGE 2 at 45° right

**Correct Output**:
{{
    "current_location": "Living room center near coffee table",
    "instruction_sequence": "Living room(current) → Kitchen doorway → Kitchen interior → Refrigerator area",
    "subtask_destination": "Kitchen doorway threshold",
    "subtask_instruction": "Turn toward doorway, navigate around coffee table edge, move to doorway",
    "planning_hints": "Currently facing TV wall with coffee table ahead. Kitchen doorway visible at angle in IMAGE 2. Strategy: Rotate to face doorway, move forward along coffee table's right edge to avoid collision, proceed toward doorway threshold until door frame is clearly visible.",
    "completion_criteria": "Door frame fills view, coffee table visible behind, kitchen interior partially visible through doorway",
    "reasoning": "(1) IMAGE 1 shows coffee table blocking direct path, IMAGE 2 reveals kitchen doorway. (2) Target doorway as first waypoint - visible landmark before entering kitchen. (3) Action plan: rotate toward doorway, navigate around coffee table obstacle, advance to threshold. (4) Verified: position → actions → criteria all reference coffee table and doorway landmarks."
}}

## Example 2: Hallway to Bedroom Task  
**Global Instruction**: "Walk down the hallway, enter the bedroom on the right, and stop beside the bed"
**Current Observation**: Agent in hallway start, bedroom door visible in IMAGE 2 at 45° right

**Correct Output**:
{{
    "current_location": "Hallway start near entrance",
    "instruction_sequence": "Hallway entrance(current) → Bedroom doorway on right → Bedroom interior → Bed area",
    "subtask_destination": "Bedroom doorway on the right",
    "subtask_instruction": "Turn toward bedroom door, move along hallway wall, stop at doorway",
    "planning_hints": "Starting at hallway entrance. Bedroom doorway visible in IMAGE 2 along corridor wall. Strategy: Rotate to face bedroom door direction, advance forward while keeping corridor wall as guide, monitor door frame approaching, stop when door threshold reached.",
    "completion_criteria": "Door frame centered in view, hallway wall visible beside doorway, bedroom interior partially visible",
    "reasoning": "(1) IMAGE 2 shows bedroom door along hallway, corridor extends beyond. (2) Select bedroom doorway as waypoint - clear landmark for entry point. (3) Actions: rotate toward door, follow wall guide, stop at threshold. (4) Checked: all descriptions reference door/wall/hallway landmarks, no abstract directions."
}}

# Key Principles
- **Actions Available**: {action_space}
- **Action Verbs**: turn toward, rotate to face, move to, navigate around, advance along, proceed to
- **First-Person Planning**: Always start from IMAGE 1 (what's directly ahead)
- **Landmark Priority**: Reference visible objects over abstract directions
- **Completion Criteria**: Use object proximity and spatial relationships (near X, between Y and Z, facing toward W)
- **Conciseness**: current_location (1 line), subtask_instruction (1 line with 3-5 actions)
"""


# Verification and Replanning prompt - combined module for subtask completion verification and next subtask generation
VERIFICATION_REPLANNING_PROMPT = """You are the verification and replanning module for Vision-Language Navigation. Your role is to verify subtask completion and plan the next action.

# Overall Navigation Task Instruction (Global Task)
{instruction}

# Current Subtask Context (from previous planning)
**Subtask Destination**: {subtask_destination}
**Subtask Instruction**: {subtask_instruction}
**Completion Criteria**: {completion_criteria}

# Current Visual Observations (8 Directional Views)
{direction_names}

IMAGE 1: Front view (0°) - What the agent directly faces (PRIMARY view for planning)
IMAGE 2: Front-Right view (45°) - Diagonal front-right direction  
IMAGE 3: Right view (90°) - Direct right side
IMAGE 4: Back-Right view (135°) - Diagonal back-right direction
IMAGE 5: Back view (180°) - Behind the agent
IMAGE 6: Back-Left view (225°) - Diagonal back-left direction
IMAGE 7: Left view (270°) - Direct left side
IMAGE 8: Front-Left view (315°) - Diagonal front-left direction

# Task Requirements

Analyze current observations to verify subtask completion, then output appropriate response:
- **If completed**: Set is_completed=true, generate NEXT subtask for global goal
- **If not completed**: Set is_completed=false, keep same destination, refine instruction based on current view

**Core Requirements:**
1. **Spatial Analysis**: Identify key landmarks, obstacles, and spatial relationships from current observations
2. **Start planning from IMAGE 1 (current front view)** - this is your first-person perspective
3. **Action-Oriented Planning**: Use concrete actions - turn toward [landmark], move to [object], navigate around [obstacle], rotate to face [target]
4. **Landmark-Based Reference**: Anchor all descriptions to visible objects/landmarks, avoid abstract directions
5. **Output Style**:
   - current_location: Concise position relative to 1-2 key landmarks
   - subtask_destination: Brief target description with landmark reference
   - subtask_instruction: Action sequence starting from IMAGE 1 (3-5 actions max)
   - planning_hints: Detailed navigation strategy from first-person perspective
   - completion_criteria: 3 object-based constraints (NO left/right/front/back - use "near [object]", "between [A] and [B]", "facing [landmark]")

**Output Quality**: Provide concise guidance that captures key spatial relationships without over-specification.

# Output Format (Strict JSON)

**MUST output ONLY valid JSON - no markdown, no extra text**

**JSON Structure (8 Required Fields - 7 from planning + is_completed):**
{{
    "is_completed": true/false,
    "current_location": "Position summary with 1-2 nearby landmarks",
    "instruction_sequence": "Complete waypoint chain: A → B → C → D (final goal)",
    "subtask_destination": "If completed: NEXT immediate waypoint. If not completed: SAME as input",
    "subtask_instruction": "If completed: Action sequence to NEXT destination starting from IMAGE 1. If not completed: REFINED action sequence to SAME destination based on current IMAGE 1 view",
    "planning_hints": "Detailed strategy from first-person view: (1) What's ahead in IMAGE 1 + obstacles, (2) Target relative position via landmarks, (3) Navigation path with actions (turn toward, move along, rotate to face, navigate around)",
    "completion_criteria": "If completed: 3 checks for NEXT destination. If not completed: 3 checks for SAME destination (use object-based: proximity, observable landmarks, spatial relationships)",
    "reasoning": "4-part logic: (1) Spatial perception from images (especially IMAGE 1), (2) Destination choice + landmark reference, (3) Action strategy from current view, (4) Completion check: verify criteria → conclusion (is_completed status)"
}}

# Examples for Understanding

## Example 1: Subtask COMPLETED - Generate Next Subtask
**Input Context**:
- Global Instruction: "Go to kitchen and stop next to refrigerator"
- Current Subtask Destination: "Kitchen doorway entrance"
- Current Subtask Instruction: "Turn toward doorway, navigate around coffee table, move to doorway"
- Completion Criteria: "Door frame fills view, coffee table visible behind, kitchen interior partially visible"
- Current Observation: IMAGE 1 shows door frame close-up filling view, coffee table in IMAGE 5 behind

**Correct Output**:
{{
    "is_completed": true,
    "current_location": "Kitchen doorway threshold",
    "instruction_sequence": "Living room → Kitchen doorway (completed) → Kitchen interior → Refrigerator (final)",
    "subtask_destination": "Kitchen interior center",
    "subtask_instruction": "Move forward through doorway, advance toward kitchen center",
    "planning_hints": "IMAGE 1 shows doorway opening directly ahead with kitchen interior visible beyond. Kitchen counters and appliances visible through doorway. Strategy: Move straight forward through doorway opening, scan for refrigerator while advancing toward kitchen center, use counter edges as navigation guides.",
    "completion_criteria": "Kitchen counters surrounding, appliances visible, doorway visible behind",
    "reasoning": "(1) IMAGE 1 shows door frame close-up filling view, IMAGE 5 shows coffee table behind - confirms at doorway threshold. (2) Kitchen doorway completed, next target is kitchen interior center before approaching refrigerator. (3) Actions from current view: move forward through doorway ahead, navigate toward center using counter landmarks. (4) Completion check: Door frame fills view ✓, coffee table behind ✓, kitchen interior visible ✓. All criteria satisfied → is_completed=true."
}}

## Example 2: Subtask NOT COMPLETED - Refine Current Instruction
**Input Context**:
- Global Instruction: "Go to kitchen and stop next to refrigerator"
- Current Subtask Destination: "Kitchen doorway entrance"
- Current Subtask Instruction: "Turn toward doorway, move forward to doorway entrance"
- Completion Criteria: "Door frame fills view, coffee table visible behind, close to entrance"
- Current Observation: IMAGE 1 shows coffee table blocking direct path, IMAGE 2 shows doorway at angle, ~4m away

**Correct Output**:
{{
    "is_completed": false,
    "current_location": "Living room near coffee table",
    "instruction_sequence": "Living room (current) → Kitchen doorway → Kitchen interior → Refrigerator (final)",
    "subtask_destination": "Kitchen doorway entrance",
    "subtask_instruction": "Navigate around coffee table edge, turn toward doorway, advance to threshold",
    "planning_hints": "IMAGE 1 shows coffee table directly ahead blocking path. IMAGE 2 reveals kitchen doorway at angle to the right. Doorway appears distant. Strategy: Rotate to face coffee table edge, move along table perimeter to clear obstacle, then rotate toward doorway direction (visible in IMAGE 2) and advance forward monitoring door frame approach.",
    "completion_criteria": "Door frame fills view, coffee table visible behind, kitchen interior partially visible",
    "reasoning": "(1) IMAGE 1 shows coffee table obstacle directly in front view, IMAGE 2 reveals doorway at angle but far. (2) Still targeting kitchen doorway entrance - need to navigate around coffee table first. (3) Actions from current view: rotate to clear coffee table ahead, navigate around obstacle, turn toward doorway, advance. (4) Completion check: NOT at doorway threshold ✗, door frame not close ✗, coffee table blocks path ✗. Criteria not satisfied → is_completed=false, need to refine with obstacle avoidance."
}}

# Key Principles
- **Actions Available**: {action_space}
- **Action Verbs**: turn toward, rotate to face, move to, navigate around, advance along, proceed to
- **First-Person Planning**: Always start from IMAGE 1 (what's directly ahead)
- **Landmark Priority**: Reference visible objects over abstract directions
- **Completion Criteria**: Use object proximity and spatial relationships (near X, between Y and Z, facing toward W)
- **Completion logic**: is_completed=true only if ALL criteria satisfied, else false with same destination
- **Conciseness**: current_location (1 line), subtask_instruction (1 line with 3-5 actions)
"""

# Task completion prompt - checks if entire navigation task is completed
TASK_COMPLETION_PROMPT = """Verify if the ENTIRE navigation task is completed.

# Navigation Goal
{instruction}

# Visual Observations (8-Direction)
{direction_names}
IMAGE 1-8: Front(0°), FR(45°), Right(90°), BR(135°), Back(180°), BL(225°), Left(270°), FL(315°)

# Task
Compare observations with goal's target location → determine completion confidence.

**Confidence must match analysis and task_completed status**

# Output Format (JSON Only)

{{
    "task_completed": true/false,
    "confidence": 0.0-1.0,
    "analysis": "Check goal elements with IMAGE citations - use object/landmark references",
    "recommendation": "If incomplete: brief next actions. If complete: null"
}}

**Confidence Scale**:
- 0.9-1.0: All goal objects present, position matches via landmarks
- 0.7-0.9: Most elements match, minor position adjustment needed
- 0.5-0.7: Partial match, significant differences in landmarks
- < 0.5: Location doesn't match goal objects

**Rules**: 
- Mark completed only if confidence > 0.85
- Use object/landmark proximity for verification, avoid abstract directions
- Ensure task_completed aligns with confidence (true only if >0.85)
"""


def get_initial_planning_prompt(instruction: str, direction_names: list, action_space: str) -> str:
    """
    Get initial planning prompt
    
    Args:
        instruction: Complete navigation instruction
        direction_names: List of direction names
        action_space: Action space description (e.g., "MOVE_FORWARD (0.25m), TURN_LEFT (45°), TURN_RIGHT (45°), STOP")
        
    Returns:
        Formatted prompt string
    """
    return INITIAL_PLANNING_PROMPT.format(
        instruction=instruction,
        direction_names=', '.join(direction_names),
        action_space=action_space
    )


def get_verification_replanning_prompt(instruction: str,
                                      subtask_destination: str,
                                      subtask_instruction: str,
                                      completion_criteria: str,
                                      direction_names: list,
                                      action_space: str) -> str:
    """
    Get verification and replanning prompt
    
    Args:
        instruction: Complete navigation instruction (global goal)
        subtask_destination: Current subtask destination
        subtask_instruction: Current subtask instruction
        completion_criteria: Completion criteria
        direction_names: List of direction names
        action_space: Action space description
        
    Returns:
        Formatted prompt string
    """
    return VERIFICATION_REPLANNING_PROMPT.format(
        instruction=instruction,
        subtask_destination=subtask_destination,
        subtask_instruction=subtask_instruction,
        completion_criteria=completion_criteria,
        direction_names=', '.join(direction_names),
        action_space=action_space
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
