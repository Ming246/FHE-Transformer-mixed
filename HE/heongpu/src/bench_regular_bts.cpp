#include <heongpu/heongpu.hpp>

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr auto kScheme = heongpu::Scheme::CKKS;
constexpr double kScale = 1.0 * (1ULL << 50);
constexpr int kPWit = 60;
constexpr int kQBodyBit = 41;
constexpr int kPBit = 60;
constexpr int kPCount = 3;
constexpr size_t kQBudgetN = 65536; // sec128 chain-length reference (run at n=4096)
constexpr int kWarmupRuns = 1;
constexpr int kTimedRuns = 3;

struct RunConfig
{
    size_t n = 4096;
    heongpu::sec_level_type sec = heongpu::sec_level_type::none;
    size_t q_budget_n = kQBudgetN;
    bool sweep_chain = false;
    int target_recovered = 5;
    bool galois_host = false;
};

struct BenchCase
{
    int ctos;
    int stoc;
    int taylor;
};

struct BenchResult
{
    BenchCase cfg;
    int q_size = 0;
    int max_depth = 0;
    int depth_before = 0;
    int depth_after = 0;
    int bts_consumed_depth = 0;
    int recovered_depth = 0;
    double time_ms = 0.0;
    std::string status;
};

std::vector<int> q_bit_sizes_sec128_budget(size_t budget_n, int p_count,
                                           int p_bit, int q_body_bit)
{
    const int total_budget = heongpu::heongpu_128bit_std_parms(budget_n);
    const int p_bits = p_count * p_bit;
    const int q_budget = total_budget - p_bits;
    if (q_budget < kPWit)
    {
        throw std::runtime_error(
            "sec128 bit budget too small for Q head prime at n=" +
            std::to_string(budget_n));
    }

    std::vector<int> q_bits;
    q_bits.push_back(kPWit);
    int used = kPWit;
    while (used + q_body_bit <= q_budget)
    {
        q_bits.push_back(q_body_bit);
        used += q_body_bit;
    }
    return q_bits;
}

RunConfig parse_args(int argc, char** argv)
{
    RunConfig cfg;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc)
        {
            cfg.n = static_cast<size_t>(std::stoul(argv[++i]));
        }
        else if (std::strcmp(argv[i], "--sec-none") == 0)
        {
            cfg.sec = heongpu::sec_level_type::none;
        }
        else if (std::strcmp(argv[i], "--sec128") == 0)
        {
            cfg.sec = heongpu::sec_level_type::sec128;
        }
        else if (std::strcmp(argv[i], "--q-budget-n") == 0 && i + 1 < argc)
        {
            cfg.q_budget_n = static_cast<size_t>(std::stoul(argv[++i]));
        }
        else if (std::strcmp(argv[i], "--sweep-chain") == 0)
        {
            cfg.sweep_chain = true;
        }
        else if (std::strcmp(argv[i], "--target-recovered") == 0 &&
                 i + 1 < argc)
        {
            cfg.target_recovered = std::stoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--galois-host") == 0)
        {
            cfg.galois_host = true;
        }
    }
    return cfg;
}

std::vector<int> p_bit_sizes()
{
    return std::vector<int>(kPCount, kPBit);
}

void drop_to_max_depth(heongpu::HEArithmeticOperator<kScheme>& ops,
                       heongpu::Ciphertext<kScheme>& ct, int q_size)
{
    const int target_depth = q_size - 1;
    while (ct.depth() < target_depth)
    {
        ops.mod_drop_inplace(ct);
    }
}

double time_bootstrap_ms(
    heongpu::HEArithmeticOperator<kScheme>& ops,
    heongpu::Ciphertext<kScheme>& ct,
    heongpu::Galoiskey<kScheme>& galois_key,
    heongpu::Relinkey<kScheme>& relin_key)
{
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    auto out = ops.regular_bootstrapping(ct, galois_key, relin_key);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    ct = std::move(out);
    return static_cast<double>(ms);
}

BenchResult run_case(
    size_t poly_modulus_degree,
    heongpu::HEContext<kScheme> context,
    heongpu::HEEncoder<kScheme>& encoder,
    heongpu::HEEncryptor<kScheme>& encryptor,
    heongpu::HEKeyGenerator<kScheme>& keygen,
    heongpu::Secretkey<kScheme>& secret_key,
    heongpu::Relinkey<kScheme>& relin_key,
    std::map<std::pair<int, int>, heongpu::Galoiskey<kScheme>>& galois_cache,
    const BenchCase& cfg,
    bool galois_host)
{
    BenchResult result;
    result.cfg = cfg;
    result.q_size = context->get_ciphertext_modulus_count();
    result.max_depth = result.q_size - 1;

    try
    {
        heongpu::HEArithmeticOperator<kScheme> ops(context, encoder);
        heongpu::BootstrappingConfig boot_config(cfg.ctos, cfg.stoc, cfg.taylor,
                                                 false);
        ops.generate_bootstrapping_params(
            kScale, boot_config,
            heongpu::arithmetic_bootstrapping_type::REGULAR_BOOTSTRAPPING);

        const auto gk_key = std::make_pair(cfg.ctos, cfg.stoc);
        if (galois_cache.find(gk_key) == galois_cache.end())
        {
            if (!galois_cache.empty())
            {
                galois_cache.clear();
            }
            std::cout << "  generating Galois keys for CtoS=" << cfg.ctos
                      << " StoC=" << cfg.stoc << " ..." << std::endl;
            std::vector<int> key_index = ops.bootstrapping_key_indexs();
            heongpu::Galoiskey<kScheme> galois_key(context, key_index);
            if (galois_host)
            {
                keygen.generate_galois_key(
                    galois_key, secret_key,
                    heongpu::ExecutionOptions().set_storage_type(
                        heongpu::storage_type::HOST));
            }
            else
            {
                keygen.generate_galois_key(galois_key, secret_key);
            }
            galois_cache.emplace(gk_key, std::move(galois_key));
        }
        auto& galois_key = galois_cache.at(gk_key);

        const int slot_count = static_cast<int>(poly_modulus_degree / 2);
        std::vector<Complex64> message(slot_count,
                                         Complex64(0.2, 0.4));

        heongpu::Plaintext<kScheme> plain(context);
        encoder.encode(plain, message, kScale);

        heongpu::Ciphertext<kScheme> ct(context);
        encryptor.encrypt(ct, plain);
        drop_to_max_depth(ops, ct, result.q_size);
        result.depth_before = ct.depth();

        if (ct.level() != 0)
        {
            throw std::runtime_error(
                "bootstrap input must be at max depth (level==0)");
        }

        for (int i = 0; i < kWarmupRuns; ++i)
        {
            heongpu::Ciphertext<kScheme> warm_ct(context);
            encryptor.encrypt(warm_ct, plain);
            drop_to_max_depth(ops, warm_ct, result.q_size);
            auto warm_out =
                ops.regular_bootstrapping(warm_ct, galois_key, relin_key);
            (void)warm_out;
        }

        double total_ms = 0.0;
        for (int i = 0; i < kTimedRuns; ++i)
        {
            heongpu::Ciphertext<kScheme> timed_ct(context);
            encryptor.encrypt(timed_ct, plain);
            drop_to_max_depth(ops, timed_ct, result.q_size);
            total_ms += time_bootstrap_ms(ops, timed_ct, galois_key, relin_key);
            if (i == 0)
            {
                result.depth_after = timed_ct.depth();
            }
        }
        result.time_ms = total_ms / kTimedRuns;

        result.bts_consumed_depth = result.depth_after;
        result.recovered_depth =
            result.max_depth - result.bts_consumed_depth;
        result.status = "ok";
    }
    catch (const std::exception& ex)
    {
        result.status = std::string("error: ") + ex.what();
    }

    return result;
}

int sum_bits(const std::vector<int>& bits)
{
    int s = 0;
    for (int b : bits)
    {
        s += b;
    }
    return s;
}

void run_chain_sweep(const RunConfig& run_cfg)
{
    constexpr BenchCase kFixedCfg{3, 3, 11};
    const auto p_bits = p_bit_sizes();
    auto q_bits =
        q_bit_sizes_sec128_budget(run_cfg.q_budget_n, kPCount, kPBit, kQBodyBit);

    std::cout << "HEonGPU REGULAR_BOOTSTRAPPING chain sweep (v1)\n";
    std::cout << "boot_config=(3,3,11) n=" << run_cfg.n
              << " sec="
              << (run_cfg.sec == heongpu::sec_level_type::sec128 ? "sec128"
                                                                 : "none")
              << " scale=2^50\n";
    std::cout << "Q head=" << kPWit << " body=" << kQBodyBit
              << " start_body_count=" << (q_bits.size() - 1)
              << " target_recovered=" << run_cfg.target_recovered << "\n";
    std::cout << "shorten: remove one " << kQBodyBit
              << "-bit prime per step until recovered=="
              << run_cfg.target_recovered << "\n";
    if (run_cfg.galois_host)
    {
        std::cout << "Galois keys: HOST storage (GPU OOM workaround)\n";
    }
    std::cout << "\n";

    std::cout << "q_body_count,Q_size,Q_bits,total_bits,max_depth,depth_after,"
                 "recovered_depth,time_ms,status\n";

    std::map<std::pair<int, int>, heongpu::Galoiskey<kScheme>> galois_cache;

    while (true)
    {
        const int q_body_count = static_cast<int>(q_bits.size()) - 1;
        const int q_sum = sum_bits(q_bits);
        const int p_sum = sum_bits(p_bits);

        std::cout << "chain q_body=" << q_body_count << " Q_size="
                  << q_bits.size() << " ..." << std::endl;

        auto context = heongpu::GenHEContext<kScheme>(run_cfg.sec);
        context->set_poly_modulus_degree(run_cfg.n);
        context->set_coeff_modulus_bit_sizes(q_bits, p_bits);
        heongpu::MemoryPoolConfig pool_config;
        pool_config.initial_device_fraction = 60.0f;
        pool_config.max_device_fraction = 99.0f;
        context->generate(pool_config);

        heongpu::HEKeyGenerator<kScheme> keygen(context);
        heongpu::Secretkey<kScheme> secret_key(context, 16);
        keygen.generate_secret_key(secret_key);

        heongpu::Publickey<kScheme> public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::Relinkey<kScheme> relin_key(context);
        keygen.generate_relin_key(relin_key, secret_key);

        heongpu::HEEncoder<kScheme> encoder(context);
        heongpu::HEEncryptor<kScheme> encryptor(context, public_key);

        const BenchResult r =
            run_case(run_cfg.n, context, encoder, encryptor, keygen,
                     secret_key, relin_key, galois_cache, kFixedCfg,
                     run_cfg.galois_host);

        std::cout << q_body_count << ',' << r.q_size << ',' << q_sum << ','
                  << (q_sum + p_sum) << ',' << r.max_depth << ','
                  << r.depth_after << ',' << r.recovered_depth << ','
                  << std::fixed << std::setprecision(3) << r.time_ms << ','
                  << r.status << '\n';

        if (r.status != "ok")
        {
            break;
        }
        if (r.recovered_depth <= run_cfg.target_recovered)
        {
            break;
        }
        if (q_bits.size() <= 2)
        {
            std::cout << "stop: cannot shorten chain further (Q_size<=2)\n";
            break;
        }

        q_bits.pop_back();
    }
}

} // namespace

int main(int argc, char** argv)
{
    const RunConfig run_cfg = parse_args(argc, argv);
    if (run_cfg.sweep_chain)
    {
        run_chain_sweep(run_cfg);
        return EXIT_SUCCESS;
    }
    const std::vector<BenchCase> cases = {
        {3, 3, 8},
        {3, 3, 9},
        {3, 3, 10},
        {3, 3, 11},
        {3, 3, 12},
        {3, 3, 13},
        {4, 4, 12},
        {4, 4, 13},
        {4, 4, 14},
        {5, 5, 13},
        {5, 5, 14},
        {5, 5, 15},
    };

    const auto q_bits =
        q_bit_sizes_sec128_budget(run_cfg.q_budget_n, kPCount, kPBit, kQBodyBit);
    const auto p_bits = p_bit_sizes();

    int q_sum = 0;
    for (int b : q_bits)
    {
        q_sum += b;
    }
    int p_sum = 0;
    for (int b : p_bits)
    {
        p_sum += b;
    }

    std::cout << "HEonGPU REGULAR_BOOTSTRAPPING benchmark (v1)\n";
    std::cout << "n=" << run_cfg.n
              << " sec="
              << (run_cfg.sec == heongpu::sec_level_type::sec128 ? "sec128"
                                                                 : "none")
              << " less_key_mode=false scale=2^50\n";
    std::cout << "Q chain: 60," << kQBodyBit << "... sized by sec128@n="
              << run_cfg.q_budget_n << "\n";
    std::cout << "Q_size=" << q_bits.size() << " Q_bits=" << q_sum
              << " P_bits=" << p_sum
              << " total_bits=" << (q_sum + p_sum)
              << " sec128_budget@n=" << run_cfg.q_budget_n << "="
              << heongpu::heongpu_128bit_std_parms(run_cfg.q_budget_n)
              << " sec128_budget@run_n="
              << heongpu::heongpu_128bit_std_parms(run_cfg.n) << "\n";
    if (run_cfg.galois_host)
    {
        std::cout << "Galois keys: HOST storage (GPU OOM workaround)\n";
    }
    std::cout << "recovered_depth = max_depth - depth_after_bts, "
                 "max_depth=Q_size-1\n\n";

    auto context = heongpu::GenHEContext<kScheme>(run_cfg.sec);
    context->set_poly_modulus_degree(run_cfg.n);
    context->set_coeff_modulus_bit_sizes(q_bits, p_bits);
    heongpu::MemoryPoolConfig pool_config;
    pool_config.initial_device_fraction = 60.0f;
    pool_config.max_device_fraction = 99.0f;
    context->generate(pool_config);

    heongpu::HEKeyGenerator<kScheme> keygen(context);
    heongpu::Secretkey<kScheme> secret_key(context, 16);
    keygen.generate_secret_key(secret_key);

    heongpu::Publickey<kScheme> public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    heongpu::Relinkey<kScheme> relin_key(context);
    keygen.generate_relin_key(relin_key, secret_key);

    heongpu::HEEncoder<kScheme> encoder(context);
    heongpu::HEEncryptor<kScheme> encryptor(context, public_key);

    std::map<std::pair<int, int>, heongpu::Galoiskey<kScheme>> galois_cache;

    std::cout << "CtoS,StoC,taylor,Q_size,max_depth,depth_before,depth_after,"
                 "bts_consumed_depth,recovered_depth,time_ms,status\n";

    for (const auto& cfg : cases)
    {
        std::cout << "case (" << cfg.ctos << "," << cfg.stoc << ","
                  << cfg.taylor << ") ..." << std::endl;
        const BenchResult r = run_case(run_cfg.n, context, encoder, encryptor, keygen,
                                       secret_key, relin_key, galois_cache,
                                       cfg, run_cfg.galois_host);
        std::cout << cfg.ctos << ',' << cfg.stoc << ',' << cfg.taylor << ','
                  << r.q_size << ',' << r.max_depth << ',' << r.depth_before
                  << ',' << r.depth_after << ',' << r.bts_consumed_depth << ','
                  << r.recovered_depth << ',' << std::fixed
                  << std::setprecision(3) << r.time_ms << ',' << r.status
                  << '\n';
    }

    return EXIT_SUCCESS;
}
