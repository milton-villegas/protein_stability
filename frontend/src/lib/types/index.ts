export interface FactorsResponse {
	factors: Record<string, string[]>;
	stock_concs: Record<string, number>;
	per_level_concs: Record<string, Record<string, Record<string, number>>>;
	total_combinations: number;
	plates_required: number;
}

export interface AvailableFactors {
	factors: Record<string, string>;
	categorical_factors: string[];
	factor_categories?: { name: string; factors: string[] }[];
}

export interface DesignTypes {
	design_types: Record<string, DesignTypeInfo>;
	resolution_options: string[];
	ccd_type_options: string[];
	d_optimal_model_options: string[];
	default_center_points: number;
	min_sample_size: number;
	max_sample_size: number;
}

export interface DesignTypeInfo {
	display_name: string;
	min_factors: number;
	max_factors: number | null;
	supports_categorical: boolean;
	requires_pydoe3: boolean;
	requires_smt: boolean;
	description: string;
	parameters: string[];
}

export interface DesignGenerateResponse {
	design_points: Record<string, any>[];
	total_runs: number;
	plates_required: number;
	warnings: string[];
}

export interface UploadResponse {
	columns: string[];
	potential_responses: string[];
	factor_columns: string[];
	preview_rows: Record<string, any>[];
	total_rows: number;
}

export interface AnalysisConfig {
	factor_columns: string[];
	categorical_factors: string[];
	numeric_factors: string[];
	data_shape: number[];
}

export interface ProjectInfo {
	name: string;
	has_design: boolean;
	has_results: boolean;
	factors_count: number;
	design_runs: number | null;
}

export interface ResponseConfig {
	name: string;
	direction: 'maximize' | 'minimize';
	min?: number;
	max?: number;
}

export interface PlotResponse {
	image: string;
	plot_type: string;
}
