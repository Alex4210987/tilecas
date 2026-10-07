# Reference parameter adapter

This reference preserves the original task mathematics and seed-42 model parameters, exposing those parameters as explicit immutable inputs. Floating activations use N(0,1). The KDA launcher uses this adapter for the corresponding attention task. New runs measure a fresh reference under the common three-warmup/five-timing protocol.
