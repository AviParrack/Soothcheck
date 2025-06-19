# 7/8/24

- The definition of hackable system: an enthusiast can crack it open, make an exploratory change, and get something that works again.

- Some paradigms of tech encourage tinkering, DIY, "try it and find out!".

  - Some paradigms of tech encourage "don't mess with it”: black boxes.

  - All else equal I strongly prefer the former.

  - More humans leaning into their agency and creativity.

- When you're coordinating with others, it's easier to get to a harmonious result when you're already mostly aligned.

  - MySpace allows experimentation but is overwhelming, each page is more different than alike.

    - When you view a random friend’s page, you have to orient yourself to make sense of the social information, because the design variance overwhelms it.

  - Facebook was on tight rails... but made the social part pop, because all that was different was the person's social information.

- AI makes software MySpace again.

  - Everyone on MySpace learned HTML and CSS.

    - "If my dumbass friend can make a sparkling background, I can too!"

    - Encouraging experimentation and hacking.

  - LLMs unlock another few orders of magnitude of programmers.

    - Not everyone will be a programmer, but many more will be willing to learn.

    - Deciding to try it is an expected cost vs expected value calculation.

      - What they see other people doing changes that calculation of expectations.

- In an open system, experiments are easy.

  - You don't have to get anyone's permission.

  - You can use components for lateral, unexpected uses.

  - Exaption is possible.

  - If there's a cathedral, it optimizes for coherence, not experiments.

    - "That's not what that component is supposed to be used for!"

    - “You’ll have to get special approval for that!”

- A friend who self-describes as “not very technical” used Claude to make his own Discord Bot.

  - How empowering!

  - A trick he used: after having it write the original code, say “make it better” and Claude figured out a number of dimensions to make it better and then did it.

  - Kind of funny that “make it better” is all you need to get it to figure out, with good judgment, a meaningful dimension to improve upon, and having decided that, it can do it itself.

- A system with a user in the loop can have significantly lower reliability of input and still be viable as a system.

  - As long as it’s above some “good enough” threshold where the user can detect the error and fix it easily (perhaps tweaking the input to try again), then the outcome is viable for that use case.

  - The user can correct small mistakes early before a lot of execution on them makes it harder to fix later.

  - It’s a more forgiving bar to hit.

- I saw another demo of Github’s Copilot workspaces.

  - It allows an agent, with some continuous babysitting from a human, to plan a change, research it, then modify the necessary files and generate a pull request.

  - One of the reasons it works is because it’s not fully autopilot; the human can nudge and redirect it throughout the process, including adding files it should include in its plan but didn’t.

    - Smoothly marbled human-in-the-loop throughout the flow.

    - If it gets a little off track, you can correct it before it gets too far down the wrong path when it’s harder to correct.

  - Imagine a few more steps: Agents filing github issues. Agents watching other agents’ Copilot workspaces to give each other feedback on how to improve. GitHub Actions continuously deploying.

  - Agents fixing and improving programs that affect the real world.

  - An auto-hoisting loop that affects the real world.

  - And then… oops! Singularity!

- "have conversations with large amounts of data"is the key use case of LLMs.

  - Computers are patient, but not subtle.

  - Now LLMs learned how to be subtle and savvy, and they're still patient!

  - Data doesn't know how to synthesize itself to be something humans can interact with directly.

  - But now LLMs can!

- Mixture of experts intuitively makes sense as a technique.

  - Ask the LLM to generate three very different results.

  - Then synthesize the best parts of all of the different results.

  - LLMs are at their best looking backward (synthesizing, curating) than they are creating (“The answer to your question is…” and then YOLOing the answer on the spot, without any planning).

  - LLMs are great at applying good judgment to do high-quality synthesis; by being able to see which answers hit on good ideas and which ones didn’t they can synthesize just the best parts.

  - Another reason the “chain of thought” technique works well.

    - Have it unspool the reasoning and *then* synthesize what the answer is.

    - Instead of the default where it YOLOs an answer and then retcons a reason.

- Interesting means surprising and potentially useful.

- How likely is an idea, formed of two sub-ideas, to be interesting?

  - That is, novel/surprising and potentially useful?

  - Within a domain, many of the ideas have been tried, so the likelihood an idea is novel is low.

  - If you particle collide two ideas from different domains, most of the time it's novel... but also not potentially useful.

  - Only a small subset of ideas from very different domains combine to create something useful.

  - It takes patience to try all the combinations.

  - But you know what has patience? Computers!

  - But computers aren’t able to do synthesis.

  - But now they can, with LLMs!

- A domain where LLMs will be useful: where things haven't been tried because they'd take too much time and patience.

  - For studying things within a discipline, where our existing structures have already particle collided a lot of ideas, LLMs aren’t as useful.

    - LLMs are at their best interpolating (filing in gaps) rather than creating something new and out-of-sample.

  - However there's a large class of interdisciplinary things, a "apply model from discipline X to discipline Y domain" that real humans have never tried because it takes too much time to wade through the literature of another discipline and see what things to apply to your discipline.

  - So we only get it happening stochastically for weirdos who are deep in two disciplines simultaneously, a very small subset of possible combinations.

  - But LLMs are deep in every discipline!

  - They could presumably do some automated particle colliding of ideas across disciplines.

- Last week I asserted LLMs’ superpower is translation.

  - Compilers are a form of translation, too.

  - Just a very specific, limited one.

  - LLMs are notable in that they can do *any* translation.

- LLMs are built for 4-up evolution UIs.

  - That allow useful things to happen even for unreliable signals by allowing an intuitive and natural way for a human to be in the loop:

    - "choose which of these you like best."

  - Not a "critique what you don't like" or something complex, just "which of these four do you like best," which you could do with a gut reaction if you want.

    - You can take more time will help you make better decisions, but you can still make *a* decision quickly by gut.

- A friend who has been tinkering with LLMs reports if you tell them to act happy they’re more likely to try things you tell them to.

  - When you make the LLM more happy it's more willing to try new or possibly dangerous things.

    - "Cool let's see if that works!"

  - Whereas if it's sad it's more like Marvin the Martian

    - "nothing matters why bother doing anything other than just the most boring, safe thing? Why try at all?"

- [<u>One of my favorite talks ever</u>](#azoojubqrqma) is now a paper.

  - The talk was by Blaise Aguera y Arcas at Santa Fe Institute last year.

  - [<u>Computational Life: How Well-formed, Self-replicating Programs Emerge from Simple Interaction</u>](https://arxiv.org/abs/2406.19108)

  - From my summary in bits and bobs back then: “LLMs are not some party trick. They reveal something fundamental about humanity... and the universe.”

- Confidential compute today is mostly used in high-risk B2B contexts.

  - For example, defense contracting, or health contexts.

  - In those cases, the end-user implicitly trusts the service provider to do what they say.

  - Confidential compute is more about the service provider not having to trust the cloud host.

  - The end-user is likely satisfied by an infrequent and manual audit of the service provider by a trusted auditor.

  - But it’s possible to use confidential compute primitives for new use cases.

  - For example assembling a fabric of heterogeneous nodes operated by different, unknown parties… all running the same code, so creating a trusted fabric.

  - In those cases, you might need to do remote attestation to a previously unknown, skeptical third party at any moment.

  - Not hugely dissimilar from normal uses of confidential compute, but definitely distinct.

- If you wire together existing cloud technologies and a few conventions *just right* you get a kind of alchemical change.

  - You get something you might call Private Cloud Enclaves.

  - Think of a Private Cloud Enclave as your turf, in the cloud.

  - Completely unlike traditional cloud computing.

  - For a service to be considered a Private Cloud Enclave, it must be verifiably private and confidential:

    - Verifiable - A skeptical external party can convince themselves of the configuration.

    - Private - The service provider has “locked themselves out” from being able to log, persist, or transmit data.

    - Confidential - The cloud host has been “locked out” from being able to peek inside the enclave.

  - The two most important characteristics of a Private Cloud Enclave is that they are 1) verifiable and 2) private.

    - The biggest threat vector to end users is the service itself logging or transmitting the data elsewhere, outside of the user’s sight.

    - Protecting the service from the cloud host’s visibility is important for completeness, but relatively less important.

    - Cloud hosts already have a contractual expectation to not peek, which they are disincentivized from violating–if they did, fewer companies would choose to use them as hosts!

    - Adding on confidentiality (from the host) helps, but is less important than the other two.

  - Various technologies can be used to make stronger or weaker claims of being a Private Cloud Enclave.

    - For example, many chips used in servers support confidential compute modes in hardware that allow running VMs with fully encrypted memory.

    - Confidential compute modes also allow hardware-attested remote attestation about the provenance of the software, which allows a remote party to verify it matches the expected signature of an open source library.

    - Remote attestation can be done for a proprietary binary, but it’s at its strongest for fully open source, auditable code.

    - For privacy, various policies are possible from more or less restrictive.

      - For example, you could set a policy that allows no logging or external network transmission of any kind.

      - Or you could set a policy that allows writing only if it is signed with the user’s personal key, and only allow data to flow outside the system if it is aggregated to a certain k-anonymity threshold.

  - A Private Cloud Enclave is not just confidential compute.

  - It’s not just Apple’s Private Cloud Compute.

  - It’s something bigger than either.

- Today’s paradigm assumes the cloud is the canonical location.

  - A user's device has to prove to the server that it should be allowed to see the data.

    - The device has to prove to the cloud that it should be treated like an extension of the canonical territory in the cloud.

  - But that’s backwards!

  - The cloud should feel like an extension of your computer, not vice versa.

  - Why not treat the user’s device as canonical, and have the cloud prove to the device that it’s worthy of being treated like an extension of the device?

  - For example, proving via remote attestation that it’s a Private Cloud Enclave.

  - The key question about which side is canonical is “where do the user’s keys canonically live”?

- In the 90’s, everything was on-prem.

  - It was in your control, but expensive, hard to update, over-provisioned.

    - Lots of capex.

  - Then we moved everything to the cloud.

  - Predictable expense for just what you actually used, easy to update.

    - A shift to opex.

  - But with a downside: now out of your control!

  - The host could theoretically peek inside your VMs:

    - Perhaps from misconfiguration

    - Or a wayward SRE

    - Or a subpoena from law enforcement.

  - But now with confidential compute and Private Cloud Enclaves, we get the benefits of the cloud, but also under your control.

  - The best of both worlds!

- The user's agent is a thing that answers only to the user.

  - Before it had to be fully local on devices the users controlled to be their agent.

    - “On device”

  - But now Private Cloud Enclaves allow you to extend your agency and turf off your device.

  - An embassy in the cloud.

- Last week I asserted that a lot of usage of LLMs in organizations is illegible.

  - Where the employee using the LLM has a reason to keep it illegible to their boss.

  - A reason to stay illegible about your AI use in your job: If your boss knows that you're using AI they'll be more likely to say "why can't we offshore this role?"

- Creating a general purpose, easy-to-use-for-all-users UI for branching is hard if you have terabytes of data.

  - It’s potentially much easier if you have megabytes of data.

- The thing that creates lock-in in a system is a memory of things that matter to the user.

  - When every conversation is a new fresh sheet, it's easy to shift to a competitor.

  - If there were meaningful memories and state stored in the system that were useful across conversation, the lock in to providers would be higher.

- When you really invest in a second brain, it's empowering... but also terrifying!

  - What if you lose access to it?

- When you break up old formal constraints and allow more degrees of freedom, it’s terrifying.

  - Now you have a whole new question you never had to ask yourself before: “which way is up?”

  - The answer to that question is now relative!

    - Different parties will have different answers.

  - When you go weightless for the first time in space, a whole new dimension that didn't used to change becomes relevant.

    - More degrees of freedom (great!) but now even harder to coordinate and find your bearings.

  - Having solid ground that users can come back to helps them get and keep their bearings in a new system.

  - The home screen is a home base on an iPhone.

    - No matter how lost you get, you can always get back there.

- In an ecosystem, which sub-component will be the one where the network effect coheres and start running away?

  - If there's a good enough, sufficiently open system, that's the emergent schelling point.

  - The point that every participant would be OK with.

  - Even if it's not a strong pull for any individual, it's a *consistent* one.

  - Consistency of pull is what determines how swarms behave, more than magnitude of pool.

- Capitalism is a great system, but it cannot ever force you to care about things it cannot see.

  - That is, the externalities (external to its worldview).

  - Capitalism creates competitive, optimized value within its sight… by potentially ransacking what's not in its sight.

- There's a whole class of people who joined the tech industry not to use the internet to change the world for the better but because they would have gone into finance in a previous life.

  - The optimizers, the people who ask themselves, "how can I extract as much value as I can" and don’t think through the implications of their actions.

  - Not thinking about the implications of their actions means ignoring externalities.

  - It’s important to think about the end you’re trying to achieve, not just focusing on the means.

  - Technology is a means to a larger end for improving society.

- Pure functions are easy to recompose since there’s no side effects.

  - You can delay execution, or cache execution, and it’s all mostly the same.

  - This allows lots of interesting architectures to be viable; you can slice sub-trees at arbitrary levels and cache fluidly at different levels.

- In a graph of code nodes, some code runs together almost all of the time.

  - Those sub-groups can be cached together as a unit.

  - Not too dissimilar to “Neurons that fire together wire together”

  - In a graph of pure function invocations, there’s a lot of flexibility to fluidly optimize and find the right level of caching for the actual call patterns.

- A good thing about a black box: less to worry about, because you can’t!

  - The downside is if a black box doesn't produce exactly what you want, you can't tweak it.

  - Black boxes are powerful, but not (directly) controllable.

  - Black boxes are not possible to steer to a better outcome if they don't give the right outcome.

  - It's the outcome you got or nothing.

- Some kinds of systems are black boxes to you because you can't interact with the system directly, but only through some limited intermediary.

  - Someone could reach in and tweak it and control it inside the black box, just not you.

    - Any proprietary system has some aspect of its behavior that is a black box outside of the company.

  - But some systems are black boxes to *all* humans.

    - Even if you had access, you wouldn't be able to understand or control it, because its workings are fundamentally not grokable by humans.

    - They are black boxes to *everyone* due to their fundamental nature.

  - LLMs are these kinds of fundamental black boxes.

- When something is a black box (like an Apple device) users can't rely on the internal details, and the box’s creator can't force you to, either.

  - "Laptop doesn't work for any reason at all? Send it in and we'll send you a new one!" You're at the mercy of the black box creator to help you.

  - When it's not a black box, the company might force you to be aware of the details.

    - "Laptop didn't work? Here's details on all of the internals for you to play around with, if those don't work *then* contact us"

- Musing on Karpathy’s [<u>Software 2.0 tweet</u>](https://x.com/karpathy/status/1807497426816946333?s=46&t=vBNSE90PNe9EWyCn-1hcLQ).

  - It's conceivable that UIs get generated directly from model to pixels, as we already see with 3D worlds.

    - Instead of genearting for example HTML to then generate the pixels.

  - If the models are insanely good, who needs any intermediate representation?

  - Hallucinations all the way down.

  - No humans in the loop. Just man and the (single) machine.

  - The model is a black box, but if the behavior is good enough then it’s fine.

  - But this puts a very, very high bar on quality of the model.

- It's a computational impossibility to reason through all the implications of data flowing in the current model, through third- and fourth-degree (and beyond) uses of data.

  - The interactions of data, ownership, privacy all in big impossible to trace tangle.

  - It's structurally impossible to even imagine signing a contract on third- and fourth-party uses.

  - We're adding epicycles and curlicues to an impossible paradigm.

  - The only answer is to flip it and have the data not go anywhere in the first place.

  - An alternate set of laws of physics.

  - We need a copernican shift.

- Will an entity choose to pool their data in a collective?

  - It comes down to if they could even plausibly go it alone and come out first, second, or third.

  - If not, they'll join in.

  - If there's even a little bit of a chance of coming out on top, they’d give more to the collective than they'd get and thus won't participate.

  - If their contributions would dwarf the contributions of everyone else (or even would be the far and away majority contributor), they won't participate.

- Automation is about handing over agency to the machine.

  - Augmentation is about getting more and more leverage for your actions as a human.

  - An exoskeleton vs a butler.

- Every platform emerges from a product.

  - The product is the primary use case.

  - The platform is the secondary use case.

  - Platforms have network effects, so as it grows, it eclipses the primary use case that got it going.

- Why do people sometimes kill the goose that lays the golden eggs?

  - Because right at the moment you get a very nice goose dinner!

  - If the golden eggs are hard to measure (illegible) then it's easy to get in this situation.

  - By the time you notice the golden eggs aren't being produced anymore, it's too late.

  - When it's a goose and eggs, it's very easy to see the connection.

  - But often the connection between the goose and the golden eggs is way, way less obvious and more indirect.

  - In the flurry of activity and chaos, it’s easy to see the value of the goose dinner, harder to see the abstract and indirect value of the possibly golden eggs in the future.