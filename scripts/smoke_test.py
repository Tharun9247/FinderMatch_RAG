import sys
import importlib.util

def main() -> None:
    spec = importlib.util.spec_from_file_location('app', 'project/streamlit_app.py')
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    matcher = module.FounderMatcher()
    print('rows', 0 if matcher.founders_df is None else len(matcher.founders_df))
    print('emb', 0 if matcher.embeddings_data is None else len(matcher.embeddings_data))
    results = matcher.search('healthtech AI founder in Paris', top_k=3)
    print('results', len(results))
    for r in results:
        print(r['id'], r['founder_name'], round(r['score'], 2), '|', '; '.join(r['matched_fields']))

if __name__ == '__main__':
    main()


