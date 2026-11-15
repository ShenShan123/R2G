"""CPU synthetic forward/backward checks for 6 models × 2 stages × supported tasks/views."""
from pathlib import Path
import sys,os,json,importlib,tempfile,subprocess
ROOT=Path(__file__).resolve().parents[1]
def native_fixture(view, contracts):
    import torch
    from torch_geometric.data import HeteroData

    torch.manual_seed(918)
    graph = HeteroData()
    contract = contracts[view]
    for name, width in contract.node_dims.items():
        graph[name].x = torch.randn(12, width)
        graph[name].x[0, 0] = float("nan")
    relations = []
    for relation, width in contract.edge_dims.items():
        if relation[1] == "congestion_geom":
            continue
        graph[relation].edge_index = torch.stack(
            (torch.arange(12), torch.arange(12).roll(1))
        )
        graph[relation].edge_attr = torch.randn(12, width)
        relations.append(relation)
    store = (
        graph["net"]
        if view in "BC"
        else graph[next(r for r in relations if r[1] == "net_connection")]
    )
    schema = "routed_wirelength_um" if view in "BC" else "net::routed_wirelength_um"
    values = torch.arange(1, 13).float().unsqueeze(1)
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[0] = False
    if view in "BC":
        store.y, store.y_valid_mask, store.y_schema = values, mask, [schema]
    else:
        store.edge_y, store.edge_y_mask, store.edge_y_schema = values, mask, [schema]
        store.entity_weight = torch.linspace(0.5, 1.0, 12)
    graph.view_type = view
    graph.prediction_stage = "route"
    graph.design_name = "synthetic__v01"
    graph.global_features = torch.ones(5)
    graph.message_passing_edge_types = relations
    return graph


def worker(project):
 import torch
 torch.set_num_threads(1);sys.path.insert(0,str(ROOT/"models"/project))
 import dataset,encoder,model
 contracts=encoder.VIEW_CONTRACTS if hasattr(encoder,"VIEW_CONTRACTS") else None
 if contracts is None:
  # Homogeneous encoder schema is dynamic; contract shapes are defined by hetero encoder.
  import importlib.util
  spec=importlib.util.spec_from_file_location("contract_encoder",ROOT/"models/here_model/encoder.py");e=importlib.util.module_from_spec(spec);sys.modules[spec.name]=e;spec.loader.exec_module(e);contracts=e.VIEW_CONTRACTS
 rows=[]
 with tempfile.TemporaryDirectory() as tmp:
  for stage in ["placement","route"]:
   for task in ["wirelength","congestion"]:
    for view in "BCDEF":
     if task=="congestion" and view=="E":continue
     g=native_fixture(view,contracts);g.prediction_stage=stage
     if task=="congestion":
      g["gate"].y=torch.arange(12,dtype=torch.float32).unsqueeze(1)/12;g["gate"].y_valid_mask=torch.ones(12,1,dtype=torch.bool);g["gate"].y_schema=["cell_congestion"]
     p=Path(tmp)/"heterograph.pt";torch.save(g,p);sample=dataset.load_native_graph(p,task=dataset.get_task_spec(task))
     for name in (["gat","gine","resgatedgcn"] if project=="homo_model" else ["han","hgt","rgcn"]):
      torch.manual_seed(42)
      if project=="homo_model":
       schema=dataset.schema_from_sample(sample);graph=dataset.convert_sample_to_homogeneous(sample,schema);net=model.HomogeneousGNN(schema=schema,model_name=name,hidden_dim=16,num_layers=4,head_layers=3,heads=4,dropout=0);pred=net.predict(graph,net.encode(graph),torch.arange(graph.target_y.numel()))
      else:
       backbone=model.build_model(name,view_type=view,node_types=tuple(sample.graph.node_types),edge_types=sample.message_edge_types,hidden_dim=16,num_layers=4,heads=4,dropout=0);head=model.NativeTaskHead(16,3);net=torch.nn.ModuleDict({"gnn":backbone,"head":head});pred=head.predict(sample.graph,backbone(sample.graph),sample.target_locations[0],torch.arange(1,12))
      loss=pred.square().mean();assert torch.isfinite(loss);loss.backward();grads=[p.grad for p in net.parameters() if p.grad is not None];assert grads and all(torch.isfinite(x).all() for x in grads);torch.optim.AdamW(net.parameters(),lr=.001).step();rows.append({"model":name,"stage":stage,"task":task,"view":view})
 print(json.dumps({"project":project,"passed":len(rows),"records":rows}))
if __name__=="__main__":
 if len(sys.argv)>1:worker(sys.argv[1])
 else:
  for project in ["homo_model","here_model"]:subprocess.run([sys.executable,__file__,project],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1"))
