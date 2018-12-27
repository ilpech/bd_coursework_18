 -- SELECT * FROM Network_storage.network_info;

create view network_info as
  select n.network_id, n.name as net_name, m.name as model_name, t.task_name
  from networks n, models m, task_types t
  where m.task_id = t.task_id and n.model_id = m.model_id;


create view network_epochs as
  select e.epoch_id, e.epoch_number, n.net_name, n.model_name, n.task_name
  from epochs e, network_info n
  where n.network_id = e.network_id;

create view scores_metrics as
  select s.epoch_id, m.metric_name, s.score_value
  from metrics m, scores s
  where m.metric_id = s.metric_id;

create view scores_epochs_network as
  select n.net_name, n.model_name, n.task_name,
    n.epoch_number,  s.metric_name, s.score_value
  from scores_metrics s, network_epochs n
  where n.epoch_id = s.epoch_id order by n.epoch_number;
