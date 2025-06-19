# 1\`0/2/23

- Evolution is an amazing powerful innovation force.

  - It's effectively massively parallelized guess-and-check.

  - The downside is it takes a long time and kills most of the mutations in the process.

- High-quality A/B testing infrastructure creates unreasonable amounts of value.

  - Experimentation is one of the most reliable ways to create value in an automatically-cohering way, by climbing hills.

  - But experimentation requires a lot of overhead to administer and extract insights from.

  - The cost for a team to create the right tools to sense what's happening in a given experiment is way larger than the amount of value any given experiment might create.

  - That means A/B testing is a perfect fit for making infrastructure: pay the cost once, and every user now benefits.

  - On Chrome, one of the most game-changing pieces of internal infrastructure was Finch, the system to administer and measure A/B tests in the wild. It was trajectory changing for us.

  - When you have mature A/B testing tools and infrastructure, the more spaghetti you throw at the wall, the faster you find valuable ideas. This can allow a [<u>self-hoisting feedback loop</u>](https://docs.google.com/document/d/1weXlr06Qy7sAcVarCivt7939V8ppHuF8u6hkxfpWXSg/edit).

- When you're in a snowstorm, approaching each instance as a special snowflake is a mistake.

  - What's most important is not how each individual instance precisely differs.

  - What's most important is that there will be an unending stream of pretty-similar snowflakes you need to deal with.

  - When you squint and take a more systems level view you see how each flake is unique, but in ways that don't matter that much.

  - How can you factor out things that many snow flakes seem to share, so when you deal with the next snowflake you're more likely to have already built solutions that partially help with it?

  - When you focus on the snowstorm and not the snowflake, the intuition to continually accrete useful infrastructure to deal with the kinds of snowflakes you stochastically expect to deal with becomes a no-brainer.

- Wisdom is experience + curiosity.

  - Experience is about the novelty you get hands-on experience with, the variance you can absorb as knowhow.

  - Curiosity is about your mindset, how open you are to extracting that novelty and noticing it in the first place.

  - You can control your mindset minute-to-minute better than you can control what experiences you're having.

    - You can only change what experiences you're having over longer time horizons or stochastically.

  - If you think something is a chore, it will be a chore.

    - You will be resentful (not curious) and you will not notice actual novelty in the task.

- There's a difference between dumb luck and smart luck.

  - Dumb luck: just happening into a given lucky outcome.

  - Smart luck: taking deliberate actions to optimize your exposure to serendipity so although you don't know precisely what the outcome will be, you can be confident that you will stochastically have a lucky outcome.

- With a playful mindset magic is possible.

  - The playful mindset might be called the "ludic mindset".

    - It's the mindset that this random arbitrary set of rules we all will pretend are infinitely powerful in this context.

  - The ludic mindset allows people to take risks within the context.

  - When a group is in the ludic mindset together, it makes it easier to find game-changing novel ideas.

  - However, the ludic mindset is extremely brittle.

    - Any person participating who conspicuously shows they *aren’t* taking the game seriously makes the people who are taking it seriously feel like chumps, and the whole thing collapses.

- The value of experimentation rises when downside cost is minimized.

  - Experimentation allows you to find new viable ideas, even if you don't understand *why* they work.

  - If an experiment might kill you, you won't do experiments.

  - But if you reduce the downside cost of a failed experiments, then the cost becomes primarily only opportunity cost...

  - ...but the upside remains!

  - You can reduce the downside cost of experimentation by for example introducing fire breaks between components (so an explosion in one part won't affect others), or in organizations by having the ludic mindset.

- There's a difference between financial services and software services.

  - Financial services inherently must be priced in bps (because they have cost that scales with value due to risk).

  - Software services typically should be priced based on the amount of value they create for a user (how much value does the user get compared to if they didn't use this software).

  - Bps-based pricing puts users in a cost-minimization mindset.

    - Users seek to minimize costs either by negotiating on costs hard, or by minimizing their use ("am I really getting \$0.23 of value out of this invoice being generated?)

    - In any case, the user is constantly in a cost-minimization mindset, which is a transactional mindset.

  - Software services, if priced properly, incentivize people to invest in and heavily use the tools in ways that create value for them.

    - That is, users are encouraged to lean on storing data in them so they become more useful to them, building their workflows around them.

    - The more that a user bases their workflows around a tool, the more they're building a little moat around themselves and the tool, making it require more work to redo it all to go to another solution.

    - You get more of a value-creation mindset, "is this the service I want to partner with for the long term".

  - If you mix financial services and software services, the cost-minimization user mindset will taint all of it.

    - For very small users, the "just one simple price, we get paid when you get paid" is simple and works well, but as users grow this tension will have a more negative effect.

  - When looking at the core value a service offers, if you split it up into financial services and software services per job-to-be-done, you can figure out which sub-components to approach differently.

  - Properly compartmentalizing these effects in how you commercialize your products can have a transformative effect on a business.

- Try to avoid making uninteresting mistakes.

  - It's OK to run into a pointy object every so often--that's what happens when you're extending the frontier of your ability.

  - But don't run into the *same* pointy objects!

- I'd rather have good execution on a great curve than great execution on a good curve.

  - The curve is the internal dynamics of the solution/context (e.g. network effects, PMF).

  - Once found the curve is almost exogenous to the execution.

  - Execution can only change the results above or below the curve by a few percentage points.

  - Finding a great curve (compounding loop) is a discontinuously great improvement.

  - However, executing is the thing that people have the most control over, and also has the shortest feedback loops.

  - Execution is not the end, it's just the (fundamentally required to cause impact) *means*.

  - But over time any means will be treated more and more as if they are the ends, absorbing more and more of the focus and effort.

- When you're focusing on executing, any doubt makes you a worse executor.

  - That can lead to behavior where people intuitively ignore potentially disconfirming evidence.

    - This can cause Goodhart's law: everyone focusing on making the metric green, not thinking as much about the amount of ground-truthed impact created.

  - But disconfirming evidence is what helps you find great curves.

- Conway's law is that the org chart will be visible externally

  - The way to erase the visibility of the org chart externally is via coordination.

  - But coordination costs scale super-linearly with the number of entities that need to coordinate.

  - If you can live with your organizational boundaries being visible externally, you can get away with much less coordination cost.

- Coherence means that the whole is greater than the sum of the parts.

  - Coherence can be cheap ("automatically cohering") or expensive to create.

  - The expensiveness of the coherence is how much effort must go into coordination to get a coherent result.

  - Coordination gets super-linearly more expensive as the number of entities that must coordinate grows, which means a non-automatically-cohering thing gets more and more expensive as it grows.

  - At some point a non-automatically cohering thing will hit an asymptote where all energy goes into coordination and none goes into net-new value creation.

  - In contrast, automatically-cohering systems instead have a compounding loop of value. The faster they go, the more value that is created, at an accelerating rate.

  - Things that make something more automatically cohering

    - A shared north star

    - Independent components with clear boundaries, where it's OK for that boundary to be visible externally (Conway's law)

    - Ecosystem (relying on third parties to create some of the value)

    - Network effects

  - If you base your value proposition on polish and coherence of your product suite, then it will not be automatically cohering.

- The Goodhart's law dynamic is like a [<u>monkey's paw</u>](https://en.wikipedia.org/wiki/The_Monkey%27s_Paw).

  - Be careful what you wish for--you just might get it!

- People tend to cling too tightly to plans.

  - People cling to plans for two reasons:

    - 1\) they are a comfort blanket of certainty.

    - 2\) They help coordinate across a large organization as a schelling point so everyone doesn't. have to continually relitigate everything.

  - However, detailed plans, especially in dynamic environments, are brittle and can be big liabilities!

    - They are extremely expensive to create and change, and also they are based on whatever naive thing you thought in the past.

    - It's hard to incorporate new things you've learned since you created the plan.

    - The more details that are pinned down in your plan, the more likely they are to be partially disconfirmed by new information, and the harder they will be to change when they are.

  - That's why it's better to have flexible plans without too much detail.

- Hype up the greatness in people around you.

  - People do their best work (growth, impact, sustainability) when they’re pushing the limits of what they can do, which requires sticking their neck out.

  - People are willing to stick their neck out when they feel confident.

  - People don’t want to look arrogant (and may have some self doubt) so tend to not want to hype *themselves* up

  - People around others can choose what to hype up.

    - This is not a “everyone and everything is great” but choosing the glimmers of greatness around you and hyping them up.

  - The other person choosing to hype up another makes it not feel self-serving, but still unlocks the confidence.

  - This unlocks way more performance and is a win win!

- For an idea to resonate and change behavior in a good way, it has to be both authentic and aspirational.

  - An alternate simple framing is "connect and redirect"

  - You embrace people / the org where they are, but point a way to a better outcome from there.

—-

Humans have an illusion of general intelligence but really it’s an ensemble of a lot of different types of modules.

Like a location mapping and loop closing model, facial recognition, catching a moving object.

We can’t even conceive of other types of modules because we don’t have them and don’t find them useful because we don’t even try the things we can’t do well

But like a magnetic orientation model is conceivable.

Built in models just feel like intuition. Just like a foggy cliff face of things that we can do well enough to feel intuitive.

These modules just feel like intuition. They all feel the same, so they all seem like they are generated by the same thing. But it's a much more craggy set of capabilities than it feels like because of that fuzziness.

We train computers to do the things that we find useful and do well.

—

RAG is a hack:

[<u>https://www.latent.space/p/llamaindex#details</u>](https://www.latent.space/p/llamaindex#details)