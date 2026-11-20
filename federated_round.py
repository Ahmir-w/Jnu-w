"""Separate-process federation: run 'client' on each Jetson, 'init/aggregate' on server.
Transfer only the parameter checkpoint and client packet (not local diagnostics).
Transport/authentication and orchestration are external to this script.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fedmr_2026.federated import FederatedCoordinator, ClientUpdate, train_client_update


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    init = sub.add_parser('init')
    init.add_argument('--embedding-dim', type=int, required=True)
    init.add_argument('--hidden-dim', type=int, default=32)
    init.add_argument('--seed', type=int, default=42)
    init.add_argument('--out', type=Path, required=True)
    client = sub.add_parser('client')
    client.add_argument('--checkpoint', type=Path, required=True)
    client.add_argument('--train-file', type=Path, required=True)
    client.add_argument('--client-id', required=True)
    client.add_argument('--out', type=Path, required=True)
    client.add_argument('--base-url', required=True, help='Use a locally hosted endpoint for on-device experiment')
    client.add_argument('--api-key', default='sk-no-key-required')
    client.add_argument('--model', required=True)
    client.add_argument('--tokenizer-name', default='meta-llama/Llama-3.1-8B-Instruct')
    client.add_argument('--embedding-model', default='all-mpnet-base-v2')
    client.add_argument('--max-conversations', type=int, default=500)
    client.add_argument('--max-pairs-per-conv', type=int, default=500)
    client.add_argument('--difficulty-threshold', type=float, default=0.3)
    client.add_argument('--clip-c', '--C', dest='C', type=float, default=1.0)
    client.add_argument('--sigma', type=float, default=0.001)
    client.add_argument('--local-epochs', type=int, default=3)
    client.add_argument('--lr', type=float, default=0.001)
    client.add_argument('--batch-size', type=int, default=64)
    client.add_argument('--seed', type=int, default=42)
    client.add_argument('--device', default='cpu')
    agg = sub.add_parser('aggregate')
    agg.add_argument('--checkpoint', type=Path, required=True)
    agg.add_argument('--updates', type=Path, nargs='+', required=True)
    agg.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix != '.npz':
        p.error('--out must end with .npz')
    if args.cmd == 'init':
        if args.embedding_dim < 1 or args.hidden_dim < 1:
            p.error('dimensions must be positive')
        FederatedCoordinator(args.embedding_dim + 5, args.hidden_dim, args.seed).save(args.out)
    elif args.cmd == 'client':
        from fedmr_2026.builder import MemoryBankBuilder
        from fedmr_2026.config import LLMConfig
        from fedmr_2026.llm_client import LLMClient
        coordinator = FederatedCoordinator.load(args.checkpoint)
        llm = LLMClient(LLMConfig(base_url=args.base_url, api_key=args.api_key, model=args.model,
                                  tokenizer_name=args.tokenizer_name, embedding_model=args.embedding_model))
        builder = MemoryBankBuilder(llm, args.max_pairs_per_conv, args.difficulty_threshold)
        builder.build_memory_bank_from_file(str(args.train_file), args.max_conversations)
        packet, diag = train_client_update(builder.memory_bank.units, coordinator.broadcast(), coordinator.round_id,
             coordinator.input_dim, coordinator.hidden_dim, args.C, args.sigma, args.local_epochs,
             args.lr, args.batch_size, args.seed + coordinator.round_id, args.device)
        # Only this packet is shared. Configuration scalars and count are NOT protected by noise.
        np.savez_compressed(args.out, delta=packet.delta, sample_count=packet.sample_count,
                            round_id=packet.round_id, client_id=args.client_id, C=args.C, sigma=args.sigma)
        local = {**diag, 'C': args.C, 'sigma': args.sigma, 'round_id': packet.round_id,
                 'local_epochs': args.local_epochs, 'lr': args.lr, 'batch_size': args.batch_size,
                 'formal_dp_guarantee': False}
        local_path = args.out.with_suffix('.LOCAL_ONLY.json')
        local_path.write_text(json.dumps(local, indent=2), encoding='utf8')
        print(f'Wrote {args.out}; keep {local_path} on this client, do not upload it.')
    else:
        coordinator = FederatedCoordinator.load(args.checkpoint)
        packets, ids, settings = [], set(), set()
        for path in args.updates:
            with np.load(path, allow_pickle=False) as f:
                client_id = str(f['client_id'])
                if client_id in ids:
                    raise ValueError('Duplicate client ID')
                ids.add(client_id)
                settings.add((float(f['C']), float(f['sigma'])))
                packets.append(ClientUpdate(int(f['round_id']), int(f['sample_count']), f['delta'].copy()))
        if len(settings) != 1:
            raise ValueError('Inconsistent C/sigma across clients')
        C, sigma = next(iter(settings))
        if not np.isfinite([C,sigma]).all() or C <= 0 or sigma < 0:
            raise ValueError('Invalid C/sigma')
        report = coordinator.aggregate(packets)
        coordinator.save(args.out)
        report.update(C=C, sigma=sigma, sigma_definition='absolute standard deviation',
                      formal_dp_guarantee=False, checkpoint=str(args.checkpoint))
        args.out.with_suffix('.round.json').write_text(json.dumps(report, indent=2), encoding='utf8')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
