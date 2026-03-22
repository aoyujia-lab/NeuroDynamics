function P = project_paths()
%PROJECT_PATHS Centralized path configuration for the steady_unsteady project.
%
%   P = project_paths();

config_dir = fileparts(mfilename('fullpath'));
paper_dir = fileparts(config_dir);
repo_root = fileparts(fileparts(paper_dir));

paper_name = 'steady_unsteady';

P.code.root = repo_root;
P.code.paper = paper_dir;
P.code.commonfuncs = fullfile(repo_root, 'common_funcs');
P.code.analysisfuncs = fullfile(paper_dir, 'funcs', 'analysis');
P.code.config = config_dir;

% ---- External data locations ----
data_root = 'E:\DATA\Steady-unsteady';
P.data.root = data_root;
P.data.raw = fullfile(data_root, 'raw');
P.data.roisignals = fullfile(data_root, 'roisignals_gsr');
P.data.behav = fullfile(data_root, 'Results', 'Behavior');

% ---- Output locations inside the repository ----
results_root = 'E:\DATA\Steady-unsteady\Results';
P.results.root = results_root;
P.results.cache = fullfile(P.results.root, 'cache');
P.results.fig = fullfile(P.results.root, 'figures');

ensure_dir(P.results.root);
ensure_dir(P.results.cache);
ensure_dir(P.results.fig);

addpath(P.code.commonfuncs, P.code.analysisfuncs, P.code.config);

P.project.name = paper_name;
end

function ensure_dir(path_str)
if ~exist(path_str, 'dir')
    mkdir(path_str);
end
end
