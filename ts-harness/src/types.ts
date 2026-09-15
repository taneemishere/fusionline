export interface Task {
  taskId: string;
  source: string;
  query: string;
  instruction: string;
  gtIds: string[];
  category: string;
}

export interface ToolRecord {
  toolId: string;
  name: string;
  description: string;
  searchText: string;
  documentation: string;
  category: string;
}

export interface RegistryView {
  key: string;
  tools: Map<string, ToolRecord>;
}

/** A tool-selection method: given a task and a registry view, return a
 *  ranked list of tool ids to expose to the model. */
export interface Strategy {
  name: string;
  /** optional model/variant tag recorded in eval rows */
  model?: string;
  select(task: Task, registry: RegistryView, k: number): Promise<string[]> | string[];
}
