import glob, csv, os

files = glob.glob('experiments/predictions/*_metrics.csv')
print(f'Total metrics files: {len(files)}')

results = []
for f in sorted(files):
    name = os.path.basename(f).replace('_metrics.csv', '')
    with open(f, 'r') as fp:
        reader = csv.DictReader(fp)
        fieldnames = reader.fieldnames or []
        first_col = fieldnames[0] if fieldnames else None
        
        # Check if first col is group or split or something else
        groups = {}
        for r in reader:
            g = r.get('group') or r.get('split') or r.get(first_col)
            if g:
                groups[g] = r
    
    tr = groups.get('train_sample/all', groups.get('train', {}))
    val_dedup = groups.get('validation/dedup', groups.get('val_dedup', {}))
    val_all = groups.get('validation/all', groups.get('val', {}))
    test = groups.get('test/all', groups.get('test', groups.get('official_test', {})))
    
    def parse_float(d, key, default=0.0):
        try:
            return float(d.get(key, default))
        except (ValueError, TypeError):
            return default

    results.append({
        'model': name,
        'groups_found': list(groups.keys()),
        'tr_acc': parse_float(tr, 'acc'),
        'tr_f1': parse_float(tr, 'f1'),
        'tr_sn': parse_float(tr, 'sn'),
        'tr_sp': parse_float(tr, 'sp'),
        'val_acc': parse_float(val_dedup, 'acc', parse_float(val_all, 'acc')),
        'val_f1': parse_float(val_dedup, 'f1', parse_float(val_all, 'f1')),
        'val_sn': parse_float(val_dedup, 'sn', parse_float(val_all, 'sn')),
        'val_sp': parse_float(val_dedup, 'sp', parse_float(val_all, 'sp')),
        'val_auc': parse_float(val_dedup, 'auc', parse_float(val_all, 'auc')),
        'test_acc': parse_float(test, 'acc'),
        'test_f1': parse_float(test, 'f1'),
        'test_sn': parse_float(test, 'sn'),
        'test_sp': parse_float(test, 'sp'),
        'test_auc': parse_float(test, 'auc'),
    })

print(f"{'Model':<24} | {'Tr Acc':<7} {'Tr F1':<7} | {'Val Acc':<7} {'Val F1':<7} | {'Test Acc':<8} {'Test F1':<8} {'Test Sn':<7} {'Test Sp':<7} {'Test AUC':<8}")
print("-" * 115)
for r in results:
    if r['test_acc'] == 0:
        print(f"{r['model']:<24} | (Groups: {r['groups_found'][:3]})")
    else:
        print(f"{r['model']:<24} | {r['tr_acc']*100:6.2f}% {r['tr_f1']*100:6.2f}% | {r['val_acc']*100:6.2f}% {r['val_f1']*100:6.2f}% | {r['test_acc']*100:7.2f}% {r['test_f1']*100:7.2f}% {r['test_sn']*100:6.2f}% {r['test_sp']*100:6.2f}% {r['test_auc']:7.4f}")
